FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /models/huggingface && \
    chown -R appuser:appuser /app /models/huggingface

COPY --chown=appuser:appuser recommendation_api.py keyword_master.json ./

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD python -c "import os, urllib.request; token = os.getenv('INTERNAL_API_TOKEN', ''); headers = {'X-Internal-Token': token} if token else {}; request = urllib.request.Request('http://127.0.0.1:8000/health', headers=headers); urllib.request.urlopen(request, timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "recommendation_api:app", "--host", "0.0.0.0", "--port", "8000"]
