FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    nginx \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/ /app/
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh /start.sh

RUN chmod +x /start.sh

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    gunicorn \
    transformers \
    torch \
    torchvision \
    torchaudio \
    pillow \
    librosa \
    opencv-python-headless \
    python-multipart

ENV PORT=8000
ENV WORKERS=2
ENV MAX_MODEL_MEMORY="2GB"

EXPOSE 80
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

CMD ["/start.sh"]
