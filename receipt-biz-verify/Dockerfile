FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m playwright install --with-deps chromium && \
    rm -rf /var/lib/apt/lists/*

COPY *.py ./

RUN python -c "from receipt_ocr import build_engine; build_engine(fast=False)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; token = os.getenv('INTERNAL_API_TOKEN', ''); headers = {'X-Internal-Token': token} if token else {}; request = urllib.request.Request('http://127.0.0.1:8000/health', headers=headers); urllib.request.urlopen(request, timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
