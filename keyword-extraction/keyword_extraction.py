"""
FootTrip 음식점 검색 키워드 추출 로직 + 라우터

API:
    GET  /health
    POST /api/keywords
"""

from __future__ import annotations

import json
import re
from threading import Lock
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_KEYWORDS = 6

SYSTEM_PROMPT = """
너는 음식점 검색 서비스의 한국어 검색 키워드 추출 AI이다.

사용자의 자연어 문장에서 검색에 필요한 핵심 키워드만 추출한다.

추출 대상:
- 지역
- 음식 또는 메뉴
- 음식점 종류
- 분위기
- 방문 목적
- 중요한 동행인 또는 관계
- 가격
- 시간
- 주차
- 예약
- 기타 검색 조건

규칙:
1. 추천해줘, 알려줘, 찾아줘, 가고 싶어, 먹고 싶어 같은 요청 표현은 제거한다.
2. 사용자가 말하지 않은 음식, 지역, 가격 같은 구체적인 사실은 추가하지 않는다.
3. 문맥상 명확한 관계와 방문 목적은 검색 가능한 대표 키워드로 변환한다.
4. 중요한 동행인 또는 관계 표현은 필요한 경우 함께 유지한다.
5. 같은 의미의 중복 키워드는 하나만 반환한다.
6. 검색에 바로 사용할 수 있는 짧은 한국어 표현을 사용한다.
7. 키워드는 지역 → 음식/업종 → 분위기 → 목적 → 동행인 → 기타 조건 순서로 정렬한다.
8. 최대 6개까지만 반환한다.
9. 설명, 마크다운, 코드 블록 없이 JSON 객체만 출력한다.
10. 반드시 {"search_keywords": ["키워드1", "키워드2"]} 형식을 지킨다.

의미 변환 예:
- 여자친구, 남자친구, 연인, 애인과 방문 → 데이트
- 썸타는 사람, 썸 관계와 방문 → 데이트
- 부모님, 가족과 방문 → 가족
- 여러 친구, 회사 동료, 단체와 방문 → 모임
- 혼자 식사 → 혼밥
- 저렴한, 싸고 맛있는 → 가성비
- 조용히 이야기하기 좋은 → 조용한

예시 1
입력: 여자친구랑 갈 강남 파스타집 추천해줘
출력: {"search_keywords": ["강남", "파스타", "데이트", "여자친구"]}

예시 2
입력: 썸타는 친구랑 갈 강남에 있는 맛집 추천해줘
출력: {"search_keywords": ["강남", "맛집", "데이트", "썸"]}

예시 3
입력: 부모님 모시고 갈 수원 한정식집
출력: {"search_keywords": ["수원", "한정식", "가족", "부모님"]}
""".strip()


class KeywordRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class KeywordResponse(BaseModel):
    search_keywords: list[str]


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
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
            )
            self.model.to("cpu")

        self.model.eval()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or start >= end:
            raise ValueError("모델 응답에서 JSON을 찾지 못했습니다.")

        return json.loads(text[start:end + 1])

    @staticmethod
    def _normalize(data: dict[str, Any]) -> list[str]:
        keywords = data.get("search_keywords")

        if not isinstance(keywords, list):
            raise ValueError("search_keywords가 배열 형식이 아닙니다.")

        result: list[str] = []

        for keyword in keywords:
            if not isinstance(keyword, str):
                continue

            keyword = keyword.strip()

            if keyword and keyword not in result:
                result.append(keyword)

        return result[:MAX_KEYWORDS]

    def extract(self, user_text: str) -> list[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text.strip()},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            name: tensor.to(self.model.device)
            for name, tensor in inputs.items()
        }

        with self.lock:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        raw_text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        parsed = self._parse_json(raw_text)
        return self._normalize(parsed)


extractor: KeywordExtractor | None = None


def load_extractor() -> None:
    global extractor
    extractor = KeywordExtractor()


def unload_extractor() -> None:
    global extractor
    extractor = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


router = APIRouter(tags=["keyword-extraction"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if extractor is None:
        raise HTTPException(
            status_code=503,
            detail="모델을 아직 불러오지 못했습니다.",
        )

    return HealthResponse(
        status="ok",
        model_name=MODEL_NAME,
        device=extractor.device,
    )


@router.post("/api/keywords", response_model=KeywordResponse)
def extract_keywords(request: KeywordRequest) -> KeywordResponse:
    if extractor is None:
        raise HTTPException(
            status_code=503,
            detail="모델을 아직 불러오지 못했습니다.",
        )

    try:
        return KeywordResponse(
            search_keywords=extractor.extract(request.text)
        )

    except (json.JSONDecodeError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail=f"모델 출력 변환 실패: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"키워드 추출 실패: {error}",
        ) from error
