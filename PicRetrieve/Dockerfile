# PicRetrieve — 本地相似图片检索服务
# 构建：docker build -t picretrieve .
# 运行：docker run --rm -p 8000:8000 -v ./data:/app/data picretrieve

FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev --frozen

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY app/ ./app/
COPY pyproject.toml ./

EXPOSE 8000

ENV PICRETRIEVE_MODEL_NAME=data/models/openai_clip-vit-base-patch32
ENV PICRETRIEVE_DATA_DIR=data
ENV PICRETRIEVE_IMAGE_ROOT=data/images
ENV PICRETRIEVE_DEVICE=auto

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
