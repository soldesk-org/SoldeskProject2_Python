"""FootTrip AI 음식점 추천 FastAPI.

최종 추천 규칙
1) 자연어를 지역/카테고리/메뉴/분위기/가격으로 분석
2) 상위 카테고리를 대표 세부 메뉴로 확장
3) Kakao Local API에서 실제 음식점 후보 조회
4) 분위기 키워드가 있으면 자체 리뷰 DB의 '일치 리뷰 비율' 내림차순 정렬
5) 비율이 같으면 일치 리뷰 수, 전체 리뷰 수 순으로 정렬
6) 최소 리뷰 수 미만 또는 리뷰가 없는 후보는 뒤쪽에 일반 추천으로 배치

실행 예:
    set KAKAO_REST_API_KEY=발급받은키
    set REVIEW_STATS_URL=http://localhost:8080/internal/reviews/keyword-ratios
    python -m uvicorn recommendation_api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("FOOTTRIP_MODEL_NAME", "kakaocorp/kanana-nano-2.1b-instruct")
KAKAO_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "").strip()
REVIEW_STATS_URL = os.getenv(
    "REVIEW_STATS_URL",
    "http://127.0.0.1:8080/internal/reviews/keyword-ratios",
).strip()
REQUEST_TIMEOUT = float(os.getenv("FOOTTRIP_HTTP_TIMEOUT", "10"))
MAX_CATEGORY_EXPANSIONS = int(os.getenv("FOOTTRIP_MAX_CATEGORY_EXPANSIONS", "6"))
MAX_KAKAO_QUERIES = int(os.getenv("FOOTTRIP_MAX_KAKAO_QUERIES", "12"))
MIN_REVIEW_COUNT = int(os.getenv("FOOTTRIP_MIN_REVIEW_COUNT", "3"))
DEFAULT_RESULT_SIZE = int(os.getenv("FOOTTRIP_DEFAULT_RESULT_SIZE", "10"))
MASTER_PATH = Path(__file__).with_name("keyword_master.json")

# ai.eattyway.com이 Cloudflare Tunnel로 인터넷에 그대로 노출돼 있어, 우리 웹 백엔드를 거치지 않은
# 직접 호출로 GPU LLM 추론과 카카오 로컬 API 호출량을 낭비당할 수 있다(2026-08-08 실측 확인 —
# 토큰 없이 /api/keywords가 그대로 200을 반환함). receipt-biz-verify(main.py)와 같은 방식으로
# X-Internal-Token 헤더 검증을 추가한다. 값이 비어있으면(로컬 개발) 검사를 건너뛴다(fail-open).
INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "").strip()

with MASTER_PATH.open("r", encoding="utf-8") as file:
    MASTER = json.load(file)

CATEGORIES: dict[str, list[str]] = MASTER["categories"]
ALIASES: dict[str, str] = MASTER["aliases"]
ATMOSPHERE_FALLBACK: dict[str, list[str]] = MASTER["atmosphere_fallback"]
VALID_CATEGORIES = set(CATEGORIES)

SYSTEM_PROMPT = """
너는 음식점 추천 서비스의 한국어 검색 문장 분석 AI이다.
반드시 아래 JSON 객체 하나만 출력한다.
{
  "location": [],
  "category": [],
  "menu_keywords": [],
  "atmosphere_keywords": [],
  "purpose_keywords": [],
  "price_min": null,
  "price_max": null,
  "other_keywords": []
}
규칙:
- category는 한식, 양식, 중식, 일식, 분식, 패스트푸드, 아시안, 술집, 뷔페, 카페/디저트, 그 외 중에서만 고른다.
- 초밥, 라멘, 돈카츠, 파스타처럼 구체적인 음식은 menu_keywords에 넣는다.
- 조용한, 감성, 분위기 좋은, 활기찬 같은 환경 조건은 atmosphere_keywords에 넣는다.
- 데이트, 가족, 모임, 혼밥은 purpose_keywords에 넣는다.
- 맛집, 추천, 찾아줘 같은 요청 표현은 제외한다.
- 사용자가 말하지 않은 정보를 임의로 만들지 않는다.
- 가격은 원 단위 정수로 변환한다. 2만원 이하라면 price_min=0, price_max=20000이다.
- 설명과 마크다운을 절대 출력하지 않는다.

입력: 강남에서 조용한 일식 맛집 추천해줘
출력: {"location":["강남"],"category":["일식"],"menu_keywords":[],"atmosphere_keywords":["조용한"],"purpose_keywords":[],"price_min":null,"price_max":null,"other_keywords":[]}
""".strip()

REVIEW_PROMPT = """
너는 음식점 리뷰를 검색용 표준 태그로 바꾸는 AI이다.
반드시 아래 JSON 객체 하나만 출력한다.
{"keywords":[{"keyword":"표준 키워드","sentiment":"positive 또는 negative"}]}
실제로 언급된 분위기, 목적, 맛, 서비스, 가격, 편의시설만 추출한다.
부정 표현을 반드시 구분한다.
예: 조용하고 분위기가 좋아요 -> 조용한 positive, 감성 positive
예: 조용하지 않고 너무 시끄러워요 -> 시끄러운 negative
""".strip()


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class RecommendRequest(TextRequest):
    x: float | None = Field(default=None, description="중심 경도")
    y: float | None = Field(default=None, description="중심 위도")
    radius: int | None = Field(default=None, ge=0, le=20000)
    size: int = Field(default=DEFAULT_RESULT_SIZE, ge=1, le=15)


class SearchAnalysis(BaseModel):
    location: list[str] = Field(default_factory=list)
    category: list[str] = Field(default_factory=list)
    menu_keywords: list[str] = Field(default_factory=list)
    atmosphere_keywords: list[str] = Field(default_factory=list)
    purpose_keywords: list[str] = Field(default_factory=list)
    price_min: int | None = None
    price_max: int | None = None
    other_keywords: list[str] = Field(default_factory=list)
    expanded_keywords: list[str] = Field(default_factory=list)
    kakao_queries: list[str] = Field(default_factory=list)


class ReviewKeyword(BaseModel):
    keyword: str
    sentiment: Literal["positive", "negative"]


class ReviewAnalysis(BaseModel):
    keywords: list[ReviewKeyword] = Field(default_factory=list)


class ReviewRatioRequest(BaseModel):
    place_ids: list[str]
    keywords: list[str]


class ReviewRatio(BaseModel):
    place_id: str
    total_review_count: int = 0
    matched_review_count: int = 0
    matched_review_ratio: float = 0.0
    matched_keywords: list[str] = Field(default_factory=list)


class RecommendedPlace(BaseModel):
    place_id: str
    place_name: str
    category_name: str = ""
    category_group_code: str = ""
    category_group_name: str = ""
    phone: str = ""
    address_name: str = ""
    road_address_name: str = ""
    place_url: str = ""
    x: str = ""
    y: str = ""
    distance: str = ""
    total_review_count: int = 0
    matched_review_count: int = 0
    matched_review_ratio: float = 0.0
    matched_keywords: list[str] = Field(default_factory=list)
    recommendation_type: Literal["REVIEW_RATIO_BASED", "CATEGORY_BASED"]
    reason: str


class RecommendResponse(BaseModel):
    analysis: SearchAnalysis
    recommendation_rule: str
    data_notice: str
    total_candidates: int
    recommendations: list[RecommendedPlace]


class KeywordExtractor:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float32
            )
            self.model.to("cpu")
        self.model.eval()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("모델 응답에서 JSON 객체를 찾지 못했습니다.")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _clean_list(value: Any, max_items: int = 8) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = ALIASES.get(item.strip(), item.strip())
            if normalized and normalized not in result:
                result.append(normalized)
        return result[:max_items]

    def _generate(self, system_prompt: str, user_text: str, max_tokens: int) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}
        with self.lock, torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self._parse_json(self.tokenizer.decode(generated, skip_special_tokens=True))

    def analyze_search(self, text: str) -> SearchAnalysis:
        data = self._generate(SYSTEM_PROMPT, text, 200)
        location = self._clean_list(data.get("location"))
        category = [c for c in self._clean_list(data.get("category")) if c in VALID_CATEGORIES]
        menus = self._clean_list(data.get("menu_keywords"))
        atmosphere = self._clean_list(data.get("atmosphere_keywords"))
        purpose = self._clean_list(data.get("purpose_keywords"))
        others = self._clean_list(data.get("other_keywords"))

        expanded = menus.copy()
        if not expanded:
            for name in category:
                expanded.extend(CATEGORIES.get(name, []))
        # 카테고리나 메뉴가 없고 분위기만 있을 때만 업종 fallback을 사용한다.
        if not expanded and atmosphere:
            for keyword in atmosphere:
                expanded.extend(ATMOSPHERE_FALLBACK.get(keyword, []))
        expanded = list(dict.fromkeys(expanded))[:MAX_CATEGORY_EXPANSIONS]

        locations = location or [""]
        terms = expanded or category or ["음식점"]
        queries = [f"{loc} {term}".strip() for loc in locations for term in terms]
        queries = list(dict.fromkeys(queries))[:MAX_KAKAO_QUERIES]

        return SearchAnalysis(
            location=location,
            category=category,
            menu_keywords=menus,
            atmosphere_keywords=atmosphere,
            purpose_keywords=purpose,
            price_min=data.get("price_min") if isinstance(data.get("price_min"), int) else None,
            price_max=data.get("price_max") if isinstance(data.get("price_max"), int) else None,
            other_keywords=others,
            expanded_keywords=expanded,
            kakao_queries=queries,
        )

    def analyze_review(self, text: str) -> ReviewAnalysis:
        data = self._generate(REVIEW_PROMPT, text, 150)
        output: list[ReviewKeyword] = []
        for item in data.get("keywords", []):
            if not isinstance(item, dict):
                continue
            keyword = item.get("keyword")
            sentiment = item.get("sentiment")
            if not isinstance(keyword, str) or sentiment not in {"positive", "negative"}:
                continue
            keyword = ALIASES.get(keyword.strip(), keyword.strip())
            if keyword and all(existing.keyword != keyword for existing in output):
                output.append(ReviewKeyword(keyword=keyword, sentiment=sentiment))
        return ReviewAnalysis(keywords=output[:8])


class KakaoLocalClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def search(self, query: str, request: RecommendRequest) -> list[dict[str, Any]]:
        if not KAKAO_REST_API_KEY:
            raise HTTPException(503, "환경변수 KAKAO_REST_API_KEY가 설정되지 않았습니다.")
        params: dict[str, Any] = {
            "query": query,
            "category_group_code": "FD6",
            "size": 15,
            "sort": "distance" if request.x is not None and request.y is not None else "accuracy",
        }
        if request.x is not None and request.y is not None:
            params["x"] = request.x
            params["y"] = request.y
        if request.radius is not None and request.x is not None and request.y is not None:
            params["radius"] = request.radius

        response = await self.client.get(
            KAKAO_URL,
            params=params,
            headers={"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"},
        )
        if response.status_code != 200:
            raise HTTPException(
                502,
                f"카카오 Local API 호출 실패({response.status_code}): {response.text[:300]}",
            )
        payload = response.json()
        return payload.get("documents", []) if isinstance(payload, dict) else []

    async def search_all(self, analysis: SearchAnalysis, request: RecommendRequest) -> list[dict[str, Any]]:
        tasks = [self.search(query, request) for query in analysis.kakao_queries]
        pages = await asyncio.gather(*tasks)
        unique: dict[str, dict[str, Any]] = {}
        for page in pages:
            for place in page:
                place_id = str(place.get("id", "")).strip()
                if place_id:
                    unique.setdefault(place_id, place)
        return list(unique.values())


class ReviewStatsClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def fetch(self, place_ids: list[str], keywords: list[str]) -> dict[str, ReviewRatio]:
        if not place_ids or not keywords:
            return {}
        try:
            headers = {"X-Internal-Token": INTERNAL_API_TOKEN} if INTERNAL_API_TOKEN else {}
            response = await self.client.post(
                REVIEW_STATS_URL,
                json={"place_ids": place_ids, "keywords": keywords},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            # 리뷰 서버가 준비되지 않은 초기 개발 단계에서도 카테고리 추천은 동작시킨다.
            return {}

        items = payload.get("items", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return {}
        result: dict[str, ReviewRatio] = {}
        for item in items:
            try:
                ratio = ReviewRatio.model_validate(item)
            except Exception:
                continue
            result[ratio.place_id] = ratio
        return result


def _build_recommendations(
    candidates: list[dict[str, Any]],
    review_stats: dict[str, ReviewRatio],
    atmosphere_keywords: list[str],
    size: int,
) -> list[RecommendedPlace]:
    ranked: list[tuple[tuple[Any, ...], RecommendedPlace]] = []
    atmosphere_requested = bool(atmosphere_keywords)

    for original_index, place in enumerate(candidates):
        place_id = str(place.get("id", ""))
        stat = review_stats.get(place_id, ReviewRatio(place_id=place_id))
        reliable = stat.total_review_count >= MIN_REVIEW_COUNT
        review_based = atmosphere_requested and reliable and stat.matched_review_count > 0

        if review_based:
            rec_type: Literal["REVIEW_RATIO_BASED", "CATEGORY_BASED"] = "REVIEW_RATIO_BASED"
            percent = round(stat.matched_review_ratio * 100, 1)
            keyword_text = ", ".join(stat.matched_keywords or atmosphere_keywords)
            reason = (
                f"전체 리뷰 {stat.total_review_count}개 중 {stat.matched_review_count}개에서 "
                f"'{keyword_text}' 조건이 확인되어 일치 비율이 {percent}%입니다."
            )
        else:
            rec_type = "CATEGORY_BASED"
            if atmosphere_requested and stat.total_review_count < MIN_REVIEW_COUNT:
                reason = (
                    "요청한 분위기를 판단할 리뷰가 아직 부족하여 "
                    "지역·음식 카테고리가 일치하는 일반 후보로 추천합니다."
                )
            elif atmosphere_requested:
                reason = "지역·음식 카테고리는 일치하지만 요청한 분위기 키워드 리뷰는 확인되지 않았습니다."
            else:
                reason = "입력한 지역과 음식 카테고리에 맞는 실제 음식점 후보입니다."

        item = RecommendedPlace(
            place_id=place_id,
            place_name=str(place.get("place_name", "")),
            category_name=str(place.get("category_name", "")),
            category_group_code=str(place.get("category_group_code", "")),
            category_group_name=str(place.get("category_group_name", "")),
            phone=str(place.get("phone", "")),
            address_name=str(place.get("address_name", "")),
            road_address_name=str(place.get("road_address_name", "")),
            place_url=str(place.get("place_url", "")),
            x=str(place.get("x", "")),
            y=str(place.get("y", "")),
            distance=str(place.get("distance", "")),
            total_review_count=stat.total_review_count,
            matched_review_count=stat.matched_review_count,
            matched_review_ratio=round(stat.matched_review_ratio, 4),
            matched_keywords=stat.matched_keywords,
            recommendation_type=rec_type,
            reason=reason,
        )

        if atmosphere_requested:
            # 리뷰 기반 후보가 먼저, 그 안에서는 비율 → 일치 리뷰 수 → 전체 리뷰 수 순.
            # 리뷰가 부족한 일반 후보는 카카오 원래 순서를 유지하며 뒤에 둔다.
            sort_key = (
                0 if review_based else 1,
                -stat.matched_review_ratio if review_based else 0,
                -stat.matched_review_count if review_based else 0,
                -stat.total_review_count if review_based else 0,
                original_index,
            )
        else:
            sort_key = (original_index,)
        ranked.append((sort_key, item))

    ranked.sort(key=lambda pair: pair[0])
    return [item for _, item in ranked[:size]]


extractor: KeywordExtractor | None = None
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor, http_client
    extractor = KeywordExtractor()
    http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    yield
    await http_client.aclose()
    http_client = None
    extractor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="FootTrip AI Recommendation API", version="3.0.0", lifespan=lifespan)


@app.middleware("http")
async def verify_internal_token(request: Request, call_next):
    if INTERNAL_API_TOKEN:
        token = request.headers.get("x-internal-token")
        if token != INTERNAL_API_TOKEN:
            return JSONResponse(
                status_code=401,
                content={"error": "UNAUTHORIZED", "message": "허용되지 않은 요청입니다."},
            )
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, Any]:
    if extractor is None:
        raise HTTPException(503, "AI 모델을 아직 불러오지 못했습니다.")
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "device": extractor.device,
        "kakao_key_configured": bool(KAKAO_REST_API_KEY),
        "review_stats_url": REVIEW_STATS_URL,
        "min_review_count": MIN_REVIEW_COUNT,
    }


@app.post("/api/keywords", response_model=SearchAnalysis)
def keywords(request: TextRequest) -> SearchAnalysis:
    if extractor is None:
        raise HTTPException(503, "AI 모델을 아직 불러오지 못했습니다.")
    try:
        return extractor.analyze_search(request.text)
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(422, f"AI 출력 JSON 변환 실패: {error}") from error


@app.post("/api/review-keywords", response_model=ReviewAnalysis)
def review_keywords(request: TextRequest) -> ReviewAnalysis:
    if extractor is None:
        raise HTTPException(503, "AI 모델을 아직 불러오지 못했습니다.")
    try:
        return extractor.analyze_review(request.text)
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(422, f"AI 리뷰 분석 JSON 변환 실패: {error}") from error


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    if extractor is None or http_client is None:
        raise HTTPException(503, "추천 서버를 아직 초기화하지 못했습니다.")

    try:
        analysis = extractor.analyze_search(request.text)
        candidates = await KakaoLocalClient(http_client).search_all(analysis, request)
        if not candidates:
            return RecommendResponse(
                analysis=analysis,
                recommendation_rule="검색 조건에 맞는 카카오 음식점 후보가 없습니다.",
                data_notice="카카오 Local API 결과가 없어 추천 결과를 만들지 못했습니다.",
                total_candidates=0,
                recommendations=[],
            )

        place_ids = [str(place.get("id", "")) for place in candidates if place.get("id")]
        review_stats = await ReviewStatsClient(http_client).fetch(
            place_ids, analysis.atmosphere_keywords
        )
        recommendations = _build_recommendations(
            candidates,
            review_stats,
            analysis.atmosphere_keywords,
            request.size,
        )

        if analysis.atmosphere_keywords:
            rule = (
                "지역·음식 카테고리 후보 중 요청한 분위기 키워드가 포함된 긍정 리뷰 비율이 "
                "높은 순서로 추천합니다. 비율이 같으면 일치 리뷰 수와 전체 리뷰 수 순으로 정렬합니다."
            )
            notice = (
                f"전체 리뷰 {MIN_REVIEW_COUNT}개 미만인 음식점은 리뷰 데이터 부족으로 판단하여 "
                "리뷰 기반 후보 뒤에 일반 추천으로 배치합니다."
            )
        else:
            rule = "분위기 조건이 없어 카카오 Local API의 지역·카테고리 검색 결과 순서를 사용합니다."
            notice = "가격 정보는 카카오 Local API 응답에 포함되지 않으므로 별도 가격 데이터가 있어야 필터링할 수 있습니다."

        return RecommendResponse(
            analysis=analysis,
            recommendation_rule=rule,
            data_notice=notice,
            total_candidates=len(candidates),
            recommendations=recommendations,
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(500, f"AI 음식점 추천 실패: {error}") from error
