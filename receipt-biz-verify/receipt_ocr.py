from __future__ import annotations

import asyncio
import base64
import difflib
import json
import logging
import re

import cv2
import numpy as np
import requests
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rapidocr import RapidOCR
from rapidocr.utils.typings import ModelType, OCRVersion

import config

router = APIRouter(tags=["receipt-ocr"])
logger = logging.getLogger(__name__)


class MenuItem(BaseModel):
    name: str
    price: int


class OcrLine(BaseModel):
    text: str
    # 0~1로 정규화된 상대 좌표(원본 이미지 크기와 무관하게 프론트에서 <img> 표시 크기에 그대로
    # 곱해서 쓸 수 있도록). 2026-08-19 추가 — 인식 성공 애니메이션에서 실제로 읽은 단어 위에
    # 초록 박스를 정확히 표시하기 위함(그 전엔 스캔 라인만 훑고 실제 위치는 안 보여줬음).
    x: float
    y: float
    w: float
    h: float


class ReceiptResult(BaseModel):
    store_name: str | None
    order_datetime: str | None
    menu_items: list[MenuItem]
    total_price: int | None
    transaction_id: str | None
    ocr_lines: list[OcrLine] = []


# ---------------------------------------------------------------------------
# RapidOCR + 정규식 기반 무료 파서
# ---------------------------------------------------------------------------

STRONG_TOTAL_KEYWORDS = [
    "합계", "총액", "총 금액", "총금액", "결제금액", "받을금액", "받은금액", "합 계", "총 합계",
    "합계금액", "총구매액", "결제 금액", "승인금액",
]
AMBIGUOUS_TOTAL_KEYWORDS = ["구매금액", "판매금액", "구 매 금 액"]
TOTAL_KEYWORDS = STRONG_TOTAL_KEYWORDS + AMBIGUOUS_TOTAL_KEYWORDS
CASH_KEYWORDS = ["현금", "받은돈", "받은 돈", "받으신", "내신돈"]
CHANGE_KEYWORDS = ["거스름", "잔돈", "거스름돈"]
TAXABLE_KEYWORDS = ["과세물품", "과세 물품", "과세물건"]
VAT_KEYWORDS = ["부가세", "부가가치세"]

IGNORE_KEYWORDS = [
    "사업자", "대표", "전화", "tel", "주소", "카드", "승인", "부가세", "과세",
    "포인트", "감사합니다", "매장", "테이블", "no.", "이용해", "환영", "번호",
    "receipt", "가맹점", "단말기", "할부", "잔액", "품명", "수량", "단가", "금액",
    "거스름", "잔돈", "받은돈", "내신돈", "현금", "매장명", "매출일", "영수증",
]

DATE_PATTERN = re.compile(r"(\d{4})[.\-/년]\s?(\d{1,2})[.\-/월]\s?(\d{1,2})일?")
TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2}))?")
PRICE_PATTERN = re.compile(r"(\d{1,3}(?:[,.]\d{3})+|\d{4,9})\s*원?")
ITEM_HEADER_PATTERN = re.compile(r"^(\d{2})(?=\D)\s*(.+)")
BARCODE_PATTERN = re.compile(r"^\d{6,}")
BRACKET_LABEL_PATTERN = re.compile(r"[\[［](.{1,6}?)[\]］]\s*(.*)")
NAME_SPLIT_PATTERN = re.compile(r"^(.*?\S)\s+(\d{6,}.*)$")

NAME_LABEL_CANDIDATES = ["매장명", "상호", "상호명"]
DATETIME_LABEL_CANDIDATES = ["주문일시", "매출일시", "매출일", "승인일시", "승인일자", "결제일시", "거래일시"]
RECEIPT_ID_LABEL_CANDIDATES = ["영수증번호", "영수번호", "거래번호", "전표번호", "승인번호"]
APPROVAL_KEYWORDS = ["승인번호", "승인 번호", "approval no", "approval"]
STANDALONE_CODE_PATTERN = re.compile(r"^[\d\-\s]{8,}$")

NUMBER_TOKEN_PATTERN = re.compile(r"\d[\d,.]*\d|\d")
ITEM_TABLE_HEADER_REFERENCE = "상품명단가수량금액"


def _clean_number(text):
    return re.sub(r"[^\d]", "", text)


def _contains_any(text, keywords):
    compact = text.replace(" ", "").lower()
    return any(k.replace(" ", "").lower() in compact for k in keywords)


def _extract_last_number(line):
    matches = NUMBER_TOKEN_PATTERN.findall(line)
    return _clean_number(matches[-1]) if matches else None


def _looks_like_item_table_header(text):
    compact = text.replace(" ", "")
    ratio = difflib.SequenceMatcher(None, compact, ITEM_TABLE_HEADER_REFERENCE).ratio()
    return ratio >= 0.55


def _extract_leading_code(text):
    compact = re.sub(r"\s+", "", text)
    m = re.match(r"^([\d\-]{6,})", compact)
    return m.group(1) if m else None


def preprocess_image(image):
    if isinstance(image, str):
        data = np.fromfile(image, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif isinstance(image, (bytes, bytearray)):
        data = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        img = image

    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_engine(fast: bool = False) -> RapidOCR:
    det_model_type = ModelType.MOBILE if fast else ModelType.SERVER
    params = {
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.model_type": det_model_type,
        "Rec.lang_type": "korean",
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_type": ModelType.MOBILE,
    }
    return RapidOCR(params=params)


def run_ocr(engine: RapidOCR, image):
    result = engine(image)
    if result.boxes is None or result.txts is None:
        return []
    return list(zip(result.boxes, result.txts, result.scores))


def group_into_lines(results, y_tolerance=12):
    items = []
    for box, text, score in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({
            "text": text.strip(), "y": sum(ys) / len(ys), "x": min(xs), "conf": score,
            # 2026-08-19 추가 — 단어 단위 박스(픽셀 좌표)를 함께 들고 다닌다. 같은 줄로 묶일 때
            # 이 박스들의 합집합을 그 줄의 표시용 박스로 쓴다.
            "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys),
        })

    items.sort(key=lambda i: i["y"])

    lines = []
    current_line = []
    anchor_y = None
    for item in items:
        if anchor_y is None or abs(item["y"] - anchor_y) <= y_tolerance:
            current_line.append(item)
            if anchor_y is None:
                anchor_y = item["y"]
        else:
            lines.append(current_line)
            current_line = [item]
            anchor_y = item["y"]
    if current_line:
        lines.append(current_line)

    line_texts = []
    line_boxes = []
    for line in lines:
        line.sort(key=lambda i: i["x"])
        text = " ".join(i["text"] for i in line)
        line_texts.append(text)
        line_boxes.append({
            "x0": min(i["x0"] for i in line), "y0": min(i["y0"] for i in line),
            "x1": max(i["x1"] for i in line), "y1": max(i["y1"] for i in line),
        })
    return line_texts, line_boxes


def _format_date(d):
    return f"{d.group(1)}-{int(d.group(2)):02d}-{int(d.group(3)):02d}"


def _find_valid_time(line):
    for m in TIME_PATTERN.finditer(line):
        if 0 <= int(m.group(1)) <= 23 and 0 <= int(m.group(2)) <= 59:
            return m
    return None


def parse_datetime(lines):
    for line in lines:
        m = BRACKET_LABEL_PATTERN.search(line.strip())
        if not m:
            continue
        label, value = m.group(1).replace(" ", ""), m.group(2).strip()
        if not value:
            continue
        best_ratio = max(
            difflib.SequenceMatcher(None, label, cand).ratio() for cand in DATETIME_LABEL_CANDIDATES
        )
        if best_ratio >= 0.5:
            d = DATE_PATTERN.search(value)
            t = _find_valid_time(value)
            if d and t:
                return f"{_format_date(d)} {t.group(1)}:{t.group(2)}"
            if d:
                return _format_date(d)

    for line in lines:
        d = DATE_PATTERN.search(line)
        t = _find_valid_time(line)
        if d and t:
            return f"{_format_date(d)} {t.group(1)}:{t.group(2)}"

    date_part = None
    time_part = None
    for line in lines:
        if date_part is None:
            d = DATE_PATTERN.search(line)
            if d:
                date_part = f"{d.group(1)}-{int(d.group(2)):02d}-{int(d.group(3)):02d}"
        if time_part is None:
            t = _find_valid_time(line)
            if t:
                time_part = f"{t.group(1)}:{t.group(2)}"
    if date_part or time_part:
        return " ".join(p for p in [date_part, time_part] if p)
    return None


def parse_restaurant_name(lines):
    for line in lines[:10]:
        m = BRACKET_LABEL_PATTERN.search(line.strip())
        if not m:
            continue
        label, value = m.group(1).replace(" ", ""), m.group(2).strip()
        if not value:
            continue
        best_ratio = max(
            difflib.SequenceMatcher(None, label, cand).ratio() for cand in NAME_LABEL_CANDIDATES
        )
        if best_ratio >= 0.5:
            return value

    for line in lines[:5]:
        clean = line.strip()
        if not clean:
            continue
        if DATE_PATTERN.search(clean) or PRICE_PATTERN.search(clean):
            continue
        if ITEM_HEADER_PATTERN.match(clean) or BRACKET_LABEL_PATTERN.search(clean):
            continue
        if _contains_any(clean, IGNORE_KEYWORDS):
            continue
        if _looks_like_item_table_header(clean):
            continue
        if len(clean) < 2:
            continue
        return clean
    return None


def parse_transaction_id(lines):
    for line in lines:
        m = BRACKET_LABEL_PATTERN.search(line.strip())
        if not m:
            continue
        label, value = m.group(1).replace(" ", ""), m.group(2).strip()
        if not value:
            continue
        best_ratio = max(
            difflib.SequenceMatcher(None, label, cand).ratio() for cand in RECEIPT_ID_LABEL_CANDIDATES
        )
        if best_ratio >= 0.5:
            code = _extract_leading_code(value)
            if code:
                return code

    for line in lines:
        if _contains_any(line, APPROVAL_KEYWORDS):
            num = _extract_last_number(line)
            if num and len(num) >= 4:
                return num

    for line in reversed(lines):
        clean = line.strip()
        if _contains_any(clean, IGNORE_KEYWORDS + TOTAL_KEYWORDS + TAXABLE_KEYWORDS + VAT_KEYWORDS + CASH_KEYWORDS + CHANGE_KEYWORDS):
            continue
        if STANDALONE_CODE_PATTERN.match(clean) and len(clean.split()) <= 2:
            code = _clean_number(clean)
            if code:
                return code

    return None


def parse_total(lines):
    for line in reversed(lines):
        if _contains_any(line, STRONG_TOTAL_KEYWORDS):
            prices = PRICE_PATTERN.findall(line)
            if prices:
                return _clean_number(prices[-1])

    for line in reversed(lines):
        if _contains_any(line, AMBIGUOUS_TOTAL_KEYWORDS):
            prices = PRICE_PATTERN.findall(line)
            if prices:
                return _clean_number(prices[-1])

    cash = None
    change = None
    for line in lines:
        if _contains_any(line, CASH_KEYWORDS):
            num = _extract_last_number(line)
            if num:
                cash = int(num)
        if _contains_any(line, CHANGE_KEYWORDS):
            num = _extract_last_number(line)
            if num:
                change = int(num)
    if cash is not None and change is not None and cash >= change:
        return str(cash - change)

    taxable = None
    vat = None
    for line in lines:
        if _contains_any(line, TAXABLE_KEYWORDS):
            num = _extract_last_number(line)
            if num:
                taxable = int(num)
        if _contains_any(line, VAT_KEYWORDS):
            num = _extract_last_number(line)
            if num:
                vat = int(num)
    if taxable is not None and vat is not None:
        return str(taxable + vat)

    all_prices = []
    for line in lines:
        for p in PRICE_PATTERN.findall(line):
            all_prices.append(_clean_number(p))
    return all_prices[-1] if all_prices else None


def _is_mostly_numeric(text):
    stripped = re.sub(r"[\s,.\-]", "", text)
    if not stripped:
        return True
    digit_count = sum(c.isdigit() for c in stripped)
    return digit_count / len(stripped) > 0.6


def _split_merged_lines(lines):
    result = []
    for line in lines:
        if BARCODE_PATTERN.match(line.strip()):
            result.append(line)
            continue
        m = NAME_SPLIT_PATTERN.match(line.strip())
        if m:
            result.append(m.group(1))
            result.append(m.group(2))
        else:
            result.append(line)
    return result


def parse_menu_items(lines):
    menu = []
    pending_name = None

    for raw_line in _split_merged_lines(lines):
        line = raw_line.strip()
        if not line:
            continue

        if _contains_any(line, TOTAL_KEYWORDS):
            break
        if _contains_any(line, IGNORE_KEYWORDS):
            pending_name = None
            continue
        if _looks_like_item_table_header(line):
            pending_name = None
            continue
        if BRACKET_LABEL_PATTERN.search(line):
            pending_name = None
            continue
        if DATE_PATTERN.search(line) or TIME_PATTERN.search(line):
            continue

        tokens = line.split()
        if tokens and BARCODE_PATTERN.match(tokens[0]):
            rest = line[len(tokens[0]):].strip()
            rest_prices = list(PRICE_PATTERN.finditer(rest))
            if rest_prices:
                name_from_line = rest[: rest_prices[0].start()].strip(" -x*·:")
                if name_from_line and not _is_mostly_numeric(name_from_line):
                    price = _clean_number(rest_prices[-1].group(1))
                    menu.append({"name": name_from_line, "price": price})
                    pending_name = None
                    continue

            amount = _clean_number(tokens[-1])
            if amount and pending_name:
                menu.append({"name": pending_name, "price": amount})
            pending_name = None
            continue

        idx_match = ITEM_HEADER_PATTERN.match(line)
        body = idx_match.group(2).strip() if idx_match else line

        prices = list(PRICE_PATTERN.finditer(body))
        if prices:
            last_price = prices[-1]
            name = body[: last_price.start()].strip(" -x*·:")
            price = _clean_number(last_price.group(1))
            if name and not _is_mostly_numeric(name):
                menu.append({"name": name, "price": price})
            pending_name = None
        elif not _is_mostly_numeric(body):
            pending_name = body

    return menu


def parse_receipt_lines(lines):
    name = parse_restaurant_name(lines)
    dt = parse_datetime(lines)
    total = parse_total(lines)
    menu = parse_menu_items(lines)
    transaction_id = parse_transaction_id(lines)

    if total and len(menu) > 1:
        menu = [item for item in menu if item["price"] != total]

    # 2026-08-19 추가 — parse_total()이 합계 줄을 못 찾거나 OCR 오인식(예: "76,000"을 "9"로 잘못 읽음)
    # 때문에 실제보다 훨씬 작은 값을 반환하는 경우가 실사용 중 확인됨(메뉴는 정상 인식됐는데 결제금액만
    # "9원"처럼 나옴). 총액이 메뉴 중 가장 비싼 항목 하나보다도 작으면 명백히 잘못된 값이므로(할인이
    # 있어도 최고가 단일 항목보다 총액이 작을 수는 거의 없음) 메뉴 합계로 대체한다.
    if menu:
        menu_sum = sum(int(item["price"]) for item in menu)
        max_item_price = max(int(item["price"]) for item in menu)
        if total is None or int(total) < max_item_price:
            total = menu_sum

    return {
        "store_name": name,
        "order_datetime": dt,
        "menu_items": [{"name": item["name"], "price": int(item["price"])} for item in menu],
        "total_price": int(total) if total else None,
        "transaction_id": transaction_id,
    }


def parse_receipt(engine: RapidOCR, image, preprocess: bool = True):
    processed = preprocess_image(image) if preprocess else image
    results = run_ocr(engine, processed)
    lines, line_boxes = group_into_lines(results)
    parsed = parse_receipt_lines(lines)

    # 2026-08-19 추가 — 줄 박스(픽셀)를 0~1 상대 좌표로 정규화해서 원본 해상도와 무관하게 프론트에서
    # 바로 쓸 수 있게 한다.
    ocr_lines = []
    if hasattr(processed, "shape"):
        img_h, img_w = processed.shape[:2]
    else:
        img_w = img_h = 0
    if img_w and img_h:
        for text, box in zip(lines, line_boxes):
            if not text:
                continue
            # 2026-08-20 수정 — RapidOCR 박스 좌표는 numpy float32라 json.dumps()가 직렬화하지 못해
            # Upstage AI 보정 호출(correct_receipt_with_ai → json.dumps(draft))이 매번 예외로 실패하고
            # 있었다("메뉴 인식이 요즘 잘 안 된다"는 지적의 원인 — AI 보정 없이 무료 파서 결과로만
            # 계속 fallback되고 있었음). float()로 명시 변환해서 표준 파이썬 float으로 통일.
            ocr_lines.append({
                "text": text,
                "x": float(box["x0"] / img_w),
                "y": float(box["y0"] / img_h),
                "w": float((box["x1"] - box["x0"]) / img_w),
                "h": float((box["y1"] - box["y0"]) / img_h),
            })
    parsed["ocr_lines"] = ocr_lines
    return parsed, lines


# ---------------------------------------------------------------------------
# Upstage Information Extract 기반 AI 보조 (무료 파서 결과가 의심스러울 때만 호출)
# ---------------------------------------------------------------------------

UPSTAGE_API_URL = "https://api.upstage.ai/v1/information-extraction"
UPSTAGE_MODEL_NAME = "information-extract"

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "store_name": {
            "type": "string",
            "description": "상호명(매장명). 영수증에 보이지 않으면 절대 지어내지 말고 반드시 null.",
        },
        "order_datetime": {
            "type": "string",
            "description": "주문/결제 일시. 'YYYY-MM-DD HH:MM' 형식. 보이지 않으면 반드시 null.",
        },
        "total_price": {
            "type": "number",
            "description": (
                "고객이 실제로 최종 결제한 총액(모든 할인 반영 후, 여러 결제수단으로 "
                "나눠 냈다면 그 합계). 부가세 제외 공급가액이 아니라 실제로 낸 돈."
            ),
        },
        "transaction_id": {
            "type": "string",
            "description": (
                "중복 등록 방지용 고유 코드. 영수증에 승인번호가 여러 개 있을 수 있는데"
                "(쿠폰/할인권 승인, 실제 카드/현금 결제 승인, 포인트 적립 승인 등), 그중 "
                "실제로 돈이 오간 결제 승인번호만 사용. 포인트 적립이나 쿠폰/할인권 자체의 "
                "승인번호는 제외. 승인번호가 없으면 영수증번호/거래번호, 그것도 없으면 "
                "영수증 하단에 찍힌 바코드 번호(전체 영수증을 대표하는 코드). 개별 상품에 "
                "붙은 상품 바코드(품목 옆의 짧은 관리번호)는 여기 쓰지 말 것. 위 어느 것도 "
                "확실하지 않으면 절대 글자를 지어내지 말고 반드시 null."
            ),
        },
        "menu_items": {
            "type": "array",
            "description": "실제 구매한 상품명과 상품별 가격 목록",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                },
            },
        },
    },
}

CORRECTION_PROMPT_TEMPLATE = """아래는 이 영수증 이미지에 대해 저사양 OCR 엔진(RapidOCR)이 1차로 뽑아낸 초안이다.
이 초안에는 다음과 같은 오류가 있을 수 있다:
- 글자가 깨져서 엉뚱한 문자로 읽힘
- 상품명이 상호명으로 잘못 들어감 (또는 그 반대)
- 승인번호가 여러 개 있는 영수증에서 엉뚱한 번호(쿠폰/포인트 적립용)를 거래코드로 잘못 채택함
- 상품 바코드처럼 관련 없는 숫자를 거래코드로 잘못 채택함
- 항목을 아예 놓치거나(null), 값을 못 찾음

첨부된 실제 영수증 이미지를 직접 보고 이 초안을 검증해라. 초안이 이미지 내용과 다르면 이미지에
보이는 실제 내용으로 고치고, 초안이 맞다면 그대로 유지해라. 초안에 없거나 null인 값도 이미지에서
확인되면 채워 넣어라. 이미지로 봐도 확실하지 않은 값은 절대 지어내지 말고 null로 남겨라.

[1차 초안(draft)]
{draft_json}
"""

_PLACEHOLDER_WORDS = {"null", "none", "n/a", "na", "unknown"}
_PLACEHOLDER_STRIP_CHARS = " \t\r\n:,./-_\"'"


def _image_part(image_bytes: bytes) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:application/octet-stream;base64,{b64}"
    return {"type": "image_url", "image_url": {"url": data_url}}


def _call_extraction_api(image_bytes: bytes, api_key: str, system_prompt: str | None = None) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [_image_part(image_bytes)]})

    payload = {
        "model": UPSTAGE_MODEL_NAME,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "receipt_schema", "schema": RECEIPT_SCHEMA},
        },
    }

    response = requests.post(
        UPSTAGE_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def _clean_ai_value(value):
    if value is None:
        return None
    text = str(value).strip()
    stripped = text.strip(_PLACEHOLDER_STRIP_CHARS)
    if stripped == "" or stripped.lower() in _PLACEHOLDER_WORDS:
        return None
    if "{" in text or "}" in text:
        return None
    if any(key in text for key in ("total_price", "transaction_id", "order_datetime", "store_name", "menu_items")):
        return None
    return stripped


def _normalize_extraction(data: dict) -> dict:
    menu_items = [
        {"name": item.get("name"), "price": int(item["price"])}
        for item in data.get("menu_items", [])
        if item.get("name") and item.get("price") is not None
    ]

    total_price = data.get("total_price")
    return {
        "store_name": _clean_ai_value(data.get("store_name")),
        "order_datetime": _clean_ai_value(data.get("order_datetime")),
        "menu_items": menu_items,
        "total_price": int(total_price) if total_price is not None else None,
        "transaction_id": _clean_ai_value(data.get("transaction_id")),
    }


def correct_receipt_with_ai(image_bytes: bytes, draft: dict, api_key: str) -> dict:
    draft_json = json.dumps(draft, ensure_ascii=False, indent=2)
    prompt = CORRECTION_PROMPT_TEMPLATE.format(draft_json=draft_json)
    data = _call_extraction_api(image_bytes, api_key, system_prompt=prompt)
    return _normalize_extraction(data)


# ---------------------------------------------------------------------------
# API 라우트
# ---------------------------------------------------------------------------


class ReceiptError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 422, parsed: dict | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.parsed = parsed


def receipt_error_response(exc: ReceiptError) -> JSONResponse:
    body = {"error": exc.code, "message": exc.message}
    if exc.parsed is not None:
        body["parsed"] = exc.parsed
    return JSONResponse(status_code=exc.status_code, content=body)


# 상호명 폴백 파서(parse_restaurant_name)가 라벨을 못 찾으면 "날짜/가격/항목 형식이 아닌 첫 줄"을 그냥
# 상호명으로 추측하는데, 그 줄에 메뉴명과 깨진 바코드/번호가 섞여 들어오는 경우가 있다(예: "김치찌개
# 공000'6"). store_name이 null이 아니라는 이유만으로 "완전하다"고 보면 이런 값을 그대로 통과시키게 되어,
# 실제 상호명에는 잘 안 나오는 문자(숫자/따옴표)를 섞어 쓴 값을 의심 신호로 추가한다(2026-07-22).
_SUSPICIOUS_STORE_NAME_CHARS = re.compile(r"[0-9'\"]")


def _needs_ai_assist(parsed: dict) -> bool:
    if not parsed["transaction_id"]:
        return True
    store_name = parsed["store_name"]
    if not store_name:
        return True
    if not parsed["order_datetime"]:
        return True
    menu_names = {item["name"] for item in parsed["menu_items"]}
    if store_name in menu_names:
        return True
    if any(name and (name in store_name or store_name in name) for name in menu_names):
        return True
    if _SUSPICIOUS_STORE_NAME_CHARS.search(store_name):
        return True
    if not parsed["menu_items"] and parsed["total_price"]:
        return True
    return False


@router.post("/parse-receipt", response_model=ReceiptResult)
async def parse_receipt_endpoint(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        return receipt_error_response(ReceiptError("EMPTY_FILE", "빈 파일입니다.", status_code=400))

    engine = request.app.state.ocr_engine

    try:
        parsed, lines = await asyncio.to_thread(parse_receipt, engine, content, True)
    except ValueError as e:
        return receipt_error_response(ReceiptError("INVALID_IMAGE", str(e), status_code=400))

    ocr_lines = parsed.get("ocr_lines", [])
    if config.UPSTAGE_API_KEY and (not lines or _needs_ai_assist(parsed)):
        try:
            parsed = await asyncio.to_thread(correct_receipt_with_ai, content, parsed, config.UPSTAGE_API_KEY)
            # AI 보정은 값만 다시 뽑아서 통째로 새 dict를 만들기 때문에(_normalize_extraction),
            # 처음 RapidOCR 단계에서 뽑아둔 위치 정보(ocr_lines)는 그대로 들고 온다 — AI가 텍스트
            # 값을 고쳐도 "화면에 실제로 어디 글자가 있었는지"는 원본 OCR 위치 그대로가 맞다.
            parsed["ocr_lines"] = ocr_lines
        except Exception:
            # 업스테이지 호출이 실패해도 무료 파서 결과로는 계속 진행하지만(가용성 우선), 로그 없이
            # 조용히 넘어가면 "보정이 왜 안 됐는지" 알 방법이 없어진다(2026-07-22 실제로 겪음).
            logger.exception("Upstage 영수증 보정 호출 실패 — 무료 파서 결과로 진행합니다.")

    if not lines and not parsed["store_name"] and not parsed["menu_items"] and not parsed["total_price"]:
        return receipt_error_response(
            ReceiptError(
                "NO_TEXT_DETECTED",
                "이미지에서 텍스트를 전혀 인식하지 못했습니다. 사진을 다시 촬영해 주세요.",
                status_code=422,
            )
        )

    if not parsed["store_name"] and not parsed["menu_items"] and not parsed["total_price"]:
        return receipt_error_response(
            ReceiptError(
                "RECEIPT_PARSE_FAILED",
                "영수증 정보를 인식하지 못했습니다.",
                status_code=422,
                parsed=parsed,
            )
        )

    if not parsed["store_name"]:
        return receipt_error_response(
            ReceiptError(
                "STORE_NAME_NOT_FOUND",
                "상호명을 인식하지 못했습니다.",
                status_code=422,
                parsed=parsed,
            )
        )

    if not parsed["order_datetime"]:
        return receipt_error_response(
            ReceiptError(
                "ORDER_DATETIME_NOT_FOUND",
                "주문 일시를 인식하지 못했습니다.",
                status_code=422,
                parsed=parsed,
            )
        )

    if not parsed["transaction_id"]:
        return receipt_error_response(
            ReceiptError(
                "TRANSACTION_ID_NOT_FOUND",
                "승인번호/영수증번호/바코드 번호를 인식하지 못했습니다.",
                status_code=422,
                parsed=parsed,
            )
        )

    return parsed
