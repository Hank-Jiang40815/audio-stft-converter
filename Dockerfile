FROM python:3.9-slim

WORKDIR /app

# 安裝系統相依套件
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 複製專案檔案
COPY . .

# 安裝套件相依
RUN pip install --no-cache-dir -r requirements.txt

# 安裝套件本身
RUN pip install -e .

# 設定環境變數
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# 預設命令
CMD ["python", "-m", "audio_stft_converter"]
