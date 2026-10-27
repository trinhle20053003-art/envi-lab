# Multi-stage Dockerfile cho EnVi Lab (Hugging Face Spaces)
# Giai đoạn 1: Build Frontend React + Vite
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Giai đoạn 2: Runtime Backend Python (TensorFlow + FastAPI)
FROM python:3.11-slim
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tạo user UID 1000 theo tiêu chuẩn bảo mật của Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    KERAS_HOME=/home/user/app/backend/.keras

WORKDIR /home/user/app

# Cài đặt dependencies Python
COPY --chown=user:user backend/requirements.txt backend/
RUN pip install --no-cache-dir --user -r backend/requirements.txt

# Sao chép model artifacts, backend code và frontend dist
COPY --chown=user:user artifacts/ artifacts/
COPY --chown=user:user backend/ backend/
COPY --chown=user:user --from=frontend-builder /app/frontend/dist frontend/dist

# Cổng mặc định: 7860 (Hugging Face) hoặc tự động nhận biến $PORT (Render/Koyeb)
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-7860}"]
