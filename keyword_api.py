"""FootTrip 자연어 검색/리뷰 키워드 분석 FastAPI 서버.

실행:
    cd python
    python -m uvicorn keyword_api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_ITEMS = 6
MASTER_PATH = Path(__file__).with_name("keyword_master.json")

with MASTER_PATH.open("r", encoding="utf-8") as file:
    MASTER = json.load(file)

CATEGORIES: dict[str, list[str]] = MASTER["categories"]
ALIASES: dict[str, str] = MASTER["aliases"]
ATMOSPHERE_FALLBACK: dict[str, list[str]] = MASTER["atmosphere_fallback"]
POSITIVE_TAGS: list[str] = MASTER["review_tags"]["positive"]
NEGATIVE_TAGS: list[str] = MASTER["review_tags"]["negative"]
TAG_SENTIMENT: dict[str, str] = {tag: "positive" for tag in POSITIVE_TAGS} | {
    tag: "negative" for tag in NEGATIVE_TAGS
}

SYSTEM_PROMPT = """
너는 음식점 검색 서비스의 한국어 자연어 분석 AI이다.
사용자의 문장을 아래 JSON 구조로만 분류한다.

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
1. category에는 한식, 양식, 중식, 일식, 분식, 패스트푸드, 아시안, 술집, 뷔페, 카페/디저트, 그 외만 사용한다.
2. 초밥, 라멘, 돈카츠, 파스타처럼 구체적인 음식은 menu_keywords에 넣는다.
3. 조용한, 감성, 분위기 좋은, 활기찬은 atmosphere_keywords에 넣는다.
4. 데이트, 가족, 모임, 혼밥은 purpose_keywords에 넣는다.
5. 맛집, 추천해줘, 찾아줘 같은 요청 표현은 제외한다.
6. 사용자가 말하지 않은 지역, 음식, 가격을 임의로 추가하지 않는다.
7. 가격은 원 단위 정수로 변환한다. 만 원 이하이면 price_min=0, price_max=10000이다.
8. 설명이나 마크다운 없이 JSON 객체만 출력한다.

입력: 강남 일식 맛집 추천해줘
출력: {"location":["강남"],"category":["일식"],"menu_keywords":[],"atmosphere_keywords":[],"purpose_keywords":[],"price_min":null,"price_max":null,"other_keywords":[]}

입력: 홍대에서 조용한 분위기의 2만원 이하 라멘집
출력: {"location":["홍대"],"category":["일식"],"menu_keywords":["라멘"],"atmosphere_keywords":["조용한","감성"],"purpose_keywords":[],"price_min":0,"price_max":20000,"other_keywords":[]}
""".strip()

# 프론트 리뷰 작성 화면은 자유 텍스트가 아니라 고정된 태그 20개(긍정 10 + 부정 10) 중에서 고르는
# 형태다(2026-07-22). 모델에게 새 키워드를 "만들어내게" 하지 않고, 정해진 후보 중 리뷰 내용과 맞는 것만
# 고르게 하는 폐쇄형 분류로 바꿨다 — 자유 추출은 "양이 적어요"를 "적" 한 글자로 잘라내는 등 불안정했다
# (직접 테스트로 확인, 2026-07-22).
REVIEW_TAG_PROMPT = """
너는 음식점 리뷰 문장을 아래 "후보 태그" 중에서 실제로 해당하는 것만 골라내는 AI이다.
새로운 태그를 만들지 말고, 반드시 후보 목록에 있는 문자열 그대로만 사용한다.

후보 태그(긍정):
{positive_tags}

후보 태그(부정):
{negative_tags}

규칙:
1. 리뷰 문장이 뜻하는 바와 실제로 관련 있는 태그만 고른다. 언급되지 않은 내용은 고르지 않는다.
2. 후보 목록에 없는 새 표현은 절대 만들지 않는다.
3. 설명, 마크다운 없이 아래 JSON 객체 하나만 출력한다.
{{"matched_tags": ["후보 태그 중 해당하는 것들"]}}

입력: 사장님이 친절하시고 재료도 신선해서 좋았어요. 다만 좀 시끄러웠어요.
출력: {{"matched_tags": ["친절해요", "재료가 신선해요", "소란스러워요"]}}

입력: 양이 너무 적고 가격도 비싼데 그냥 그랬어요.
출력: {{"matched_tags": ["양이 적어요", "가격이 비싸요"]}}
""".strip().format(
    positive_tags=", ".join(POSITIVE_TAGS),
    negative_tags=", ".join(NEGATIVE_TAGS),
)


class KeywordRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class SearchAnalysis(BaseModel):
    location: list[str] = []
    category: list[str] = []
    menu_keywords: list[str] = []
    atmosphere_keywords: list[str] = []
    purpose_keywords: list[str] = []
    price_min: int | None = None
    price_max: int | None = None
    other_keywords: list[str] = []
    search_keywords: list[str] = []
    expanded_keywords: list[str] = []
    kakao_queries: list[str] = []
    recommendation_basis: str = "CATEGORY_BASED"


class ReviewKeyword(BaseModel):
    keyword: str
    sentiment: str


class ReviewAnalysis(BaseModel):
    keywords: list[ReviewKeyword]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    device: str


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
            raise ValueError("모델 응답에서 JSON을 찾지 못했습니다.")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _clean_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            item = ALIASES.get(item.strip(), item.strip())
            if item and item not in result:
                result.append(item)
        return result[:MAX_ITEMS]

    def _generate(self, system_prompt: str, user_text: str, max_tokens: int = 180) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text.strip()},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    def analyze_search(self, user_text: str) -> SearchAnalysis:
        data = self._generate(SYSTEM_PROMPT, user_text)
        location = self._clean_list(data.get("location"))
        category = [c for c in self._clean_list(data.get("category")) if c in CATEGORIES]
        menus = self._clean_list(data.get("menu_keywords"))
        atmosphere = self._clean_list(data.get("atmosphere_keywords"))
        purpose = self._clean_list(data.get("purpose_keywords"))
        others = self._clean_list(data.get("other_keywords"))

        # 사용자가 세부 메뉴를 직접 말하면 그 메뉴를 우선하고, 상위 카테고리만 말했을 때만 확장한다.
        expanded = menus.copy()
        if not expanded:
            for category_name in category:
                expanded.extend(CATEGORIES.get(category_name, []))
        expanded = list(dict.fromkeys(expanded))[:MAX_ITEMS]

        # 리뷰 데이터가 없는 초기 서비스에서 사용할 약한 업종 기반 fallback.
        if not expanded and atmosphere:
            for keyword in atmosphere:
                expanded.extend(ATMOSPHERE_FALLBACK.get(keyword, []))
        expanded = list(dict.fromkeys(expanded))[:MAX_ITEMS]

        locations = location or [""]
        query_terms = expanded or category or ["음식점"]
        kakao_queries = [f"{loc} {term}".strip() for loc in locations for term in query_terms]
        kakao_queries = list(dict.fromkeys(kakao_queries))[:12]

        search_keywords = list(dict.fromkeys(location + menus + category + atmosphere + purpose + others))
        basis = "REVIEW_READY" if atmosphere else "CATEGORY_BASED"

        return SearchAnalysis(
            location=location,
            category=category,
            menu_keywords=menus,
            atmosphere_keywords=atmosphere,
            purpose_keywords=purpose,
            price_min=data.get("price_min") if isinstance(data.get("price_min"), int) else None,
            price_max=data.get("price_max") if isinstance(data.get("price_max"), int) else None,
            other_keywords=others,
            search_keywords=search_keywords,
            expanded_keywords=expanded,
            kakao_queries=kakao_queries,
            recommendation_basis=basis,
        )

    def analyze_review(self, review_text: str) -> ReviewAnalysis:
        data = self._generate(REVIEW_TAG_PROMPT, review_text, 140)
        output: list[ReviewKeyword] = []
        for item in data.get("matched_tags", []):
            if not isinstance(item, str):
                continue
            tag = item.strip()
            sentiment = TAG_SENTIMENT.get(tag)
            # 후보 목록에 없는 태그는 모델이 지어낸 것이므로 버린다 — 고정 태그 집합만 허용.
            if sentiment is None or any(x.keyword == tag for x in output):
                continue
            output.append(ReviewKeyword(keyword=tag, sentiment=sentiment))
        return ReviewAnalysis(keywords=output)


extractor: KeywordExtractor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor
    extractor = KeywordExtractor()
    yield
    extractor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="FootTrip Search AI API", version="2.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if extractor is None:
        raise HTTPException(503, "모델을 아직 불러오지 못했습니다.")
    return HealthResponse(status="ok", model_name=MODEL_NAME, device=extractor.device)


@app.post("/api/keywords", response_model=SearchAnalysis)
def extract_keywords(request: KeywordRequest) -> SearchAnalysis:
    if extractor is None:
        raise HTTPException(503, "모델을 아직 불러오지 못했습니다.")
    try:
        return extractor.analyze_search(request.text)
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(422, f"모델 출력 변환 실패: {error}") from error
    except Exception as error:
        raise HTTPException(500, f"키워드 추출 실패: {error}") from error


@app.post("/api/review-keywords", response_model=ReviewAnalysis)
def extract_review_keywords(request: KeywordRequest) -> ReviewAnalysis:
    if extractor is None:
        raise HTTPException(503, "모델을 아직 불러오지 못했습니다.")
    try:
        return extractor.analyze_review(request.text)
    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(422, f"리뷰 분석 결과 변환 실패: {error}") from error
    except Exception as error:
        raise HTTPException(500, f"리뷰 키워드 추출 실패: {error}") from error
