from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import business_auth
import config
import receipt_ocr
from receipt_ocr import ReceiptError, build_engine, receipt_error_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ocr_engine = build_engine(fast=False)
    yield
    app.state.ocr_engine = None


app = FastAPI(title="사업자 인증 + 영수증 OCR 서버", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(business_auth.router)
app.include_router(receipt_ocr.router)


# 2026-08-08 — api.eattyway.com은 인터넷에 그대로 노출돼 있어, 악의적인 사용자가 우리 웹 백엔드를 거치지
# 않고 직접 /verify·/parse-receipt·/health를 호출해 Gemini/Upstage/국세청 API 호출량을 낭비시킬 수 있다.
# Spring 쪽(BusinessVerificationClient/ReceiptOcrClient/SystemStatusServiceImpl)이 보내는
# X-Internal-Token 헤더가 이 값과 일치할 때만 통과시킨다. /health까지 포함해 예외 없이 전부 검사한다.
# INTERNAL_API_TOKEN이 비어있으면(로컬 개발 중 아직 안 정했을 때) 검사를 건너뛴다(fail-open) — 운영(VM)
# 배포 시에는 반드시 채워야 한다.
@app.middleware("http")
async def verify_internal_token(request: Request, call_next):
    if config.INTERNAL_API_TOKEN:
        token = request.headers.get("x-internal-token")
        if token != config.INTERNAL_API_TOKEN:
            return JSONResponse(
                status_code=401,
                content={"error": "UNAUTHORIZED", "message": "허용되지 않은 요청입니다."},
            )
    return await call_next(request)


@app.exception_handler(ReceiptError)
async def receipt_error_handler(request: Request, exc: ReceiptError):
    return receipt_error_response(exc)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "OCR_PROCESSING_FAILED", "message": f"처리 중 오류가 발생했습니다: {exc}"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
