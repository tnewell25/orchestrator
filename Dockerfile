FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for fastembed (ONNX) and asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download embedding model so first request isn't slow
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY . ./orchestrator/

EXPOSE 8000

# Diagnostic startup — surface import errors before uvicorn swallows them.
# Shell form so $PORT (injected by Railway) expands at runtime.
CMD echo "[boot] PORT=$PORT DATABASE_URL=${DATABASE_URL:0:30}..." && \
    python -c "import orchestrator.main; print('[boot] imports OK', flush=True)" && \
    exec uvicorn orchestrator.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
