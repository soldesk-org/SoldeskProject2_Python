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
