import whisper
model_name = "large‑v3‑turbo"
model = whisper.load_model(model_name, download_root="./whisper_models")
print("模型下载完成")