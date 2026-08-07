"""
FootTrip 키워드 추출 서버

실행:
    python -m uvicorn main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

import keyword_extraction


@asynccontextmanager
async def lifespan(app: FastAPI):
    keyword_extraction.load_extractor()
    yield
    keyword_extraction.unload_extractor()


app = FastAPI(title="FootTrip Keyword AI API", version="1.0.0", lifespan=lifespan)

app.include_router(keyword_extraction.router)
