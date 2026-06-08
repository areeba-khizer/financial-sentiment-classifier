# Serve the fine-tuned financial sentiment API.
#
# The trained model is not baked into the image (it's large and reproducible).
# Train it locally with `python -m src.train`, then mount it at runtime:
#
#   docker build -t financial-sentiment .
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/models:/app/models:ro" financial-sentiment
#
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch keeps the image lean; install heavy deps before app code so
# Docker layer caching survives source changes.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY src ./src
COPY app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
