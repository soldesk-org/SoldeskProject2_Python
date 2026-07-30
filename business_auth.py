from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import fitz  # PyMuPDF
import requests
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from playwright.async_api import async_playwright

import config

router = APIRouter(tags=["business-auth"])


def to_image_bytes(file_bytes: bytes, filename: str, content_type: str | None) -> tuple[bytes, str]:
    is_pdf = (content_type == "application/pdf") or filename.lower().endswith(".pdf")
    if not is_pdf:
        return file_bytes, content_type or "image/png"

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        if doc.page_count == 0:
            raise ValueError("PDF에 페이지가 없습니다.")
        page = doc[0]
        zoom = 300 / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png"), "image/png"
    finally:
        doc.close()


_EXTRACTION_PROMPT = """\
다음은 한국 국세청 사업자등록증명(또는 사업자등록증) 문서 이미지입니다.
아래 JSON 스키마에 맞춰 값만 추출해서 응답하세요. 값을 찾을 수 없으면 빈 문자열("")로 두세요.
하이픈(-) 없이 숫자만 추출하세요.

{
  "business_number": "사업자등록번호 10자리, 예: 2563301857",
  "issue_number": "문서 상단의 발급번호 14자리, 예: 68786881703565",
  "representative_name": "대표자성명",
  "start_date": "개업일 8자리 YYYYMMDD",
  "company_name": "상호(법인명)",
  "address": "사업장 소재지(사업장 주소) 전체 텍스트"
}
"""


class OcrExtractionError(Exception):
    pass


def extract_business_fields(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    if not config.GEMINI_API_KEY:
        raise OcrExtractionError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")

    client = genai.Client(api_key=config.GEMINI_API_KEY)

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                _EXTRACTION_PROMPT,
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception as e:
        raise OcrExtractionError(f"OCR 호출 실패: {e}") from e

    try:
        data = json.loads(response.text)
    except (ValueError, TypeError) as e:
        raise OcrExtractionError(f"OCR 응답을 JSON으로 해석하지 못했습니다: {response.text!r}") from e

    return {
        "business_number": str(data.get("business_number", "")).strip(),
        "issue_number": str(data.get("issue_number", "")).strip(),
        "representative_name": str(data.get("representative_name", "")).strip(),
        "start_date": str(data.get("start_date", "")).strip(),
        "company_name": str(data.get("company_name", "")).strip(),
        "address": str(data.get("address", "")).strip(),
    }

NTS_BASE_URL = "https://api.odcloud.kr/api/nts-businessman/v1"
NTS_VALIDATE_URL = f"{NTS_BASE_URL}/validate"


def validate_business(businesses: list[dict]) -> dict:
    response = requests.post(
        NTS_VALIDATE_URL,
        params={"serviceKey": config.NTS_SERVICE_KEY},
        headers={"Content-Type": "application/json"},
        data=json.dumps({"businesses": businesses}, ensure_ascii=False).encode("utf-8"),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


@dataclass
class OriginCheckResult:
    found: bool
    civil_petition_name: str = ""
    issue_no_masked: str = ""
    reg_no: str = ""


def _split_issue_no(issue_no: str) -> list[str]:
    parts = issue_no.replace(" ", "").split("-")
    if len(parts) != 4:
        raise ValueError(f"발급번호 형식이 올바르지 않습니다(4-3-4-3자리 필요): {issue_no}")
    return parts


def _split_biz_no(biz_no: str) -> list[str]:
    parts = biz_no.replace(" ", "").split("-")
    if len(parts) != 3:
        raise ValueError(f"사업자등록번호 형식이 올바르지 않습니다(3-2-5자리 필요): {biz_no}")
    return parts


async def check_certificate_origin(issue_no: str, biz_no: str, headless: bool = True) -> OriginCheckResult:
    issue_parts = _split_issue_no(issue_no)
    biz_parts = _split_biz_no(biz_no)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        try:
            page = await (await browser.new_context()).new_page()

            await page.goto("https://hometax.go.kr/", wait_until="networkidle", timeout=30000)
            await page.fill('input[placeholder="검색어를 입력하세요!"]', "민원증명 원본확인")
            await page.wait_for_timeout(500)
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(2000)

            menu_links = page.locator("text=민원증명 원본 확인(수요처 조회)")
            clicked = False
            for i in range(await menu_links.count()):
                if await menu_links.nth(i).is_visible():
                    await menu_links.nth(i).click(force=True, timeout=10000)
                    clicked = True
                    break
            if not clicked:
                raise RuntimeError("'민원증명 원본 확인' 메뉴 링크를 찾지 못했습니다.")

            await page.wait_for_timeout(2000)
            await page.click("text=발급번호로 조회")
            await page.wait_for_timeout(1500)

            for i, val in enumerate(issue_parts, start=1):
                await page.fill(f"#mf_txppWframe_txtcerCvaIsnNo{i}", val)

            await page.click('label[for="mf_txppWframe_rbNtplBmanClsfCd_input_1"]')
            await page.wait_for_timeout(500)

            await page.fill("#mf_txppWframe_inputTxprBmanRgtNo1", biz_parts[0])
            await page.fill("#mf_txppWframe_inputTxprBmanRgtNo2", biz_parts[1])
            await page.fill("#mf_txppWframe_inputTxprBmanRgtNo3", biz_parts[2])

            await page.click("#mf_txppWframe_trigger1", force=True)
            await page.wait_for_timeout(3000)

            data_row = page.locator("table").filter(has_text="민원사무명").locator("tbody tr").first
            cells = data_row.locator("td")

            if await cells.count() < 3 or (await cells.nth(0).inner_text()).strip() == "":
                return OriginCheckResult(found=False)

            return OriginCheckResult(
                found=True,
                civil_petition_name=(await cells.nth(0).inner_text()).strip(),
                issue_no_masked=(await cells.nth(1).inner_text()).strip(),
                reg_no=(await cells.nth(2).inner_text()).strip(),
            )
        finally:
            await browser.close()


def _fail(status_code: int, status: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": status, "message": message},
    )


def _format_biz_no(digits: str) -> str:
    return f"{digits[0:3]}-{digits[3:5]}-{digits[5:10]}"


def _format_issue_no(digits: str) -> str:
    return f"{digits[0:4]}-{digits[4:7]}-{digits[7:11]}-{digits[11:14]}"


@router.post("/verify")
async def verify(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        return _fail(400, "NO_FILE_PROVIDED", "업로드된 파일이 비어 있습니다.")

    try:
        image_bytes, mime_type = to_image_bytes(file_bytes, file.filename or "", file.content_type)
    except Exception as e:
        return _fail(400, "FILE_PROCESSING_ERROR", f"파일을 이미지로 변환하지 못했습니다: {e}")

    try:
        fields = await asyncio.to_thread(extract_business_fields, image_bytes, mime_type)
    except OcrExtractionError as e:
        return _fail(400, "OCR_EXTRACTION_FAILED", str(e))

    biz_no_digits = fields["business_number"]
    issue_no_digits = fields["issue_number"]

    if len(biz_no_digits) != 10 or not biz_no_digits.isdigit():
        return _fail(
            400,
            "MISSING_REQUIRED_FIELDS",
            f"이미지에서 사업자등록번호(10자리)를 찾지 못했습니다. 추출값: {biz_no_digits!r}",
        )
    if len(issue_no_digits) != 14 or not issue_no_digits.isdigit():
        return _fail(
            400,
            "MISSING_REQUIRED_FIELDS",
            f"이미지에서 발급번호(14자리)를 찾지 못했습니다. 추출값: {issue_no_digits!r}",
        )

    biz_no = _format_biz_no(biz_no_digits)
    issue_no = _format_issue_no(issue_no_digits)

    try:
        origin_result = await check_certificate_origin(issue_no, biz_no)
    except Exception as e:
        return _fail(400, "ORIGIN_CHECK_ERROR", f"원본확인 조회 중 오류가 발생했습니다: {e}")

    if not origin_result.found:
        return _fail(
            400,
            "ORIGIN_NOT_FOUND",
            "발급번호/사업자등록번호로 원본 문서를 확인하지 못했습니다 (위변조 의심 또는 발급 후 90일 경과).",
        )

    try:
        validation = await asyncio.to_thread(
            validate_business,
            [
                {
                    "b_no": biz_no_digits,
                    "start_dt": fields["start_date"],
                    "p_nm": fields["representative_name"],
                    "p_nm2": "",
                    "b_nm": fields["company_name"],
                    "corp_no": "",
                    "b_sector": "",
                    "b_type": "",
                    "b_adr": fields["address"],
                }
            ],
        )
    except Exception as e:
        return _fail(400, "VALIDATION_CHECK_ERROR", f"진위확인 API 호출 중 오류가 발생했습니다: {e}")

    data_list = validation.get("data") or []
    if not data_list or data_list[0].get("valid") != "01":
        reason = data_list[0].get("valid_msg") if data_list else validation.get("msg", "알 수 없는 사유")
        return _fail(400, "VALIDATION_FAILED", f"진위확인에 실패했습니다: {reason}")

    return JSONResponse(
        status_code=200,
        content={
            "status": "SUCCESS",
            "extracted": fields,
            "origin_check": {
                "civil_petition_name": origin_result.civil_petition_name,
                "issue_no_masked": origin_result.issue_no_masked,
                "reg_no": origin_result.reg_no,
            },
            "validation": data_list[0],
        },
    )
