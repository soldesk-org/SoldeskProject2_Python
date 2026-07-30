from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
NTS_SERVICE_KEY = os.environ.get("NTS_SERVICE_KEY", "")
UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY", "")

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
