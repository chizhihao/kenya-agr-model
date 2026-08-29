FROM python:3.11-slim

WORKDIR /app

# 只安装最小系统工具：wget ca-certificates gcc编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 下载Linux静态ffmpeg二进制（不apt install ffmpeg，减少几百MB垃圾依赖）
RUN wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    && tar -xf ffmpeg-master-latest-linux64-gpl.tar.xz \
    && cp ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ \
    && cp ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ \
    && rm -rf ffmpeg-master-latest-linux64-gpl ffmpeg-master-latest-linux64-gpl.tar.xz

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8000

CMD ["python","main.py"]
