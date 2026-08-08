from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
NTS_SERVICE_KEY = os.environ.get("NTS_SERVICE_KEY", "")
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY", "")

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# api.eattyway.com이 인터넷에 그대로 노출돼 있어, CORS(브라우저에서 JS로 호출할 때만 막음)만으로는
# curl/스크립트로 직접 때리는 요청을 못 막는다(2026-08-08). Spring 백엔드가 보내는 X-Internal-Token
# 헤더와 이 값을 비교해서 서버 대 서버 호출만 허용한다. 비어있으면(로컬 개발) 검사를 건너뛴다(fail-open).
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "")
