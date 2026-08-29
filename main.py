import os
os.environ["PATH"] = os.getcwd() + os.pathsep + os.environ["PATH"]
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import json
from typing import List, AsyncIterable
import torch
import tempfile
from dotenv import load_dotenv
import whisper
from edge_tts import Communicate


# 加载.env配置文件
load_dotenv()


# ===================== 读取环境变量 =====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
SQLALCHEMY_DATABASE_URL = os.getenv("DB_URL")
SECRET_KEY = os.getenv("SECRET_KEY")


# ===================== 数据库配置 MySQL =====================
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# 用户数据表
class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True)
    phone = Column(String(20), index=True)
    password_hash = Column(String(256))


Base.metadata.create_all(bind=engine)


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120


# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========== bcrypt密码加密 ==========
def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=120)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="token无效，请重新登录",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user is None:
        raise credentials_exception
    return user


# Pydantic 请求模型
class UserRegister(BaseModel):
    username: str
    phone: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    question: str
    lang: str
    session_id: str


class ClearSessionReq(BaseModel):
    session_id: str


# ===================== 大模型 & 会话内存 =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-chat",
    temperature=0.2
)


# ---------------------- Whisper ASR 斯瓦希里语音识别 ----------------------
MODEL_FOLDER = "./whisper_models"
os.makedirs(MODEL_FOLDER, exist_ok=True)

# 从环境变量读取模型，漏写默认使用 medium
whisper_model_name = os.getenv("WHISPER_MODEL", "medium")
asr_model = whisper.load_model(whisper_model_name, download_root=MODEL_FOLDER)

# 检测GPU，有GPU就放到cuda，没有自动跑CPU
if torch.cuda.is_available():
    asr_model = asr_model.to("cuda")
    print(f"Whisper 使用GPU运行，模型：{whisper_model_name}")
else:
    print(f"未检测到GPU，Whisper使用CPU运行，模型：{whisper_model_name}")


session_store = {}
MAX_HISTORY_MSG_COUNT = 12


# 模拟RAG，暂时直接返回空，不访问向量库
async def get_rag_context(q: str):
    return ""


# ===================== 用户认证接口  =====================
@app.post("/api/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="用户名已存在")
    hashed_pwd = get_password_hash(user.password)
    new_user = DBUser(
        username=user.username,
        phone=user.phone,
        password_hash=hashed_pwd
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"code":0,"msg":"注册成功"}


@app.post("/api/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或者密码错误")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type":"bearer",
        "username": db_user.username
    }


@app.get("/api/user/info")
def get_user_info(current_user: DBUser = Depends(get_current_user)):
    return {
        "code": 0,
        "username": current_user.username,
        "phone": current_user.phone
    }


# ===================== 聊天流式接口 =====================
async def stream_generator(question: str, lang: str, session_id: str) -> AsyncIterable[str]:
    rag_context = await get_rag_context(question)
    if session_id not in session_store:
        session_store[session_id] = []
    history: List = session_store[session_id]

    system_prompt = f"""
你是Imarika肯尼亚农业助手，专门回答肯尼亚当地农作物、病虫害、种植技术问题。
优先使用知识库参考内容作答，知识库没有内容就如实说明不知道。

# 【硬性强制输出语言规则，优先级最高】
> 用户界面已经选定输出语言：{lang}
> **无论用户输入使用什么语言，你的全部回复、解释、标点，全部只能使用 {lang}，绝对禁止其他语言。**
> 不要夹杂中文、英文，完整输出目标语言。

【知识库参考上下文】
{rag_context}
"""
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(history)
    messages.append(HumanMessage(content=question))

    full_answer_chunks = []
    try:
        async for chunk in llm.astream(messages):
            piece = chunk.content
            if piece:
                full_answer_chunks.append(piece)
                yield f"data: {json.dumps({'content': piece}, ensure_ascii=True)}\n\n"
        yield f"data: {json.dumps({'end': True}, ensure_ascii=True)}\n\n"

        ai_reply = "".join(full_answer_chunks)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=ai_reply))
        if len(history) > MAX_HISTORY_MSG_COUNT:
            session_store[session_id] = history[-MAX_HISTORY_MSG_COUNT:]
    except Exception as e:
        print(f"LLM异常:{str(e).encode('utf-8','replace').decode('utf-8')}")
        yield f"data: {json.dumps({'error':'大模型调用失败'}, ensure_ascii=True)}\n\n"


@app.post("/api/chat-stream")
async def chat_stream(req: ChatRequest, _user = Depends(get_current_user)):
    return StreamingResponse(
        stream_generator(req.question, req.lang, req.session_id),
        media_type="text/event-stream"
    )


@app.post("/api/clear-session")
def clear_session(req: ClearSessionReq, _user = Depends(get_current_user)):
    if req.session_id in session_store:
        del session_store[req.session_id]
    return {"ok":True}


# 斯瓦希里语音识别接口（调试临时去掉鉴权；调试完成后加上 , _user = Depends(get_current_user)）
@app.post("/api/asr/swahili")
async def asr_swahili(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_f:
        tmp_f.write(audio_bytes)
        tmp_path = tmp_f.name
    try:
        result = asr_model.transcribe(
            tmp_path,
            language="sw",
            fp16=False
        )
        return {"text": result["text"]}
    except Exception as e:
        print(f"ASR错误: {e}")
        raise HTTPException(status_code=400, detail="音频解析失败，请重新录音")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ===================== TTS语音合成接口（edge-tts）【调试：临时取消鉴权】 =====================
@app.get("/api/tts")
async def tts_endpoint(
    text: str = Query(..., description="待合成文本"),
    lang: str = Query("en", description="语言 zh/en/sw")
):
    voice_map = {
        "zh": "zh-CN-XiaoxiaoNeural",
        "en": "en-US-AriaNeural",
        "sw": "sw-KE-ZuriNeural"
    }
    voice_name = voice_map.get(lang, "en-US-AriaNeural")
    communicate = Communicate(text, voice=voice_name, rate="+0%")

    async def audio_generator():
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            print(f"TTS生成异常: {e}")

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Content-Disposition": "inline; filename=speech.mp3"
    }
    return StreamingResponse(
        audio_generator(),
        media_type="audio/mpeg",
        headers=headers
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, reload=False)
