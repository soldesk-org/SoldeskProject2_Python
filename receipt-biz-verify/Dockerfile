# 사업자등록증 인증 + 영수증 OCR 서버(FastAPI). Project2/compose.yaml의 ocr-api 서비스가 이 파일을
# 빌드한다(2026-08-19 — compose.yaml/README는 이미 있었지만 이 Dockerfile 자체가 없어서 빌드가 항상
# 실패하던 걸 채움). RapidOCR(onnxruntime/opencv)과 사업자등록증 원본확인용 Playwright(Chromium
# headless)가 같이 필요해서 이미지가 다소 크다.
FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless가 slim 이미지에서 필요로 하는 최소 공유 라이브러리.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright 브라우저 바이너리 + 실행에 필요한 시스템 의존성을 함께 설치.
RUN playwright install --with-deps chromium

COPY . .

# RapidOCR은 처음 쓰일 때 모델 파일(onnx, 수십MB)을 modelscope.cn에서 자동 다운받아 site-packages
# 안에 저장한다 — 빌드 시점에 미리 한 번 실행해서 이미지 레이어에 구워 넣지 않으면, 컨테이너를 새로
# 띄울 때마다(재배포·재시작마다) 매번 다시 다운로드해야 해서 느려지고 외부망(modelscope.cn) 장애에도
# 취약해진다(2026-08-19 실측 확인). build_engine(fast=False)만 호출해서 모델을 내려받게 한다.
RUN python -c "from receipt_ocr import build_engine; build_engine(fast=False)"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
