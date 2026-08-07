"""web_menu_keyword_db.xlsx(README/restaurant_category/menu_keyword/keyword_alias 시트)로 keyword_master.json을
재생성한다. 손으로 쓴 카테고리 6개/별칭 몇 개짜리 초안 대신, 실제 팀 마스터 데이터(1276개 메뉴 키워드,
40개 별칭)를 그대로 반영한다. atmosphere_fallback/review_tags는 엑셀에 없는 정보라 그대로 유지/추가한다.
1회성 빌드 스크립트 — 실행: venv/Scripts/python.exe build_keyword_master.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import openpyxl

EXCEL_PATH = r"C:\Users\sdedu01\Documents\카카오톡 받은 파일\web_menu_keyword_db.xlsx"
OUT_PATH = Path(__file__).with_name("keyword_master.json")

wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)

# ---- restaurant_category ----
ws = wb["restaurant_category"]
rows = [r for r in ws.iter_rows(min_row=3, values_only=True) if r[0] not in (None, "category_id")]
category_code_to_name = {r[1]: r[2] for r in rows if r[6]}  # is_active만

# ---- menu_keyword ----
ws = wb["menu_keyword"]
rows = [r for r in ws.iter_rows(min_row=3, values_only=True) if r[0] not in (None, "keyword_id")]

cat_keywords: dict[str, list[tuple[int, str]]] = defaultdict(list)
for (keyword_id, category_code, group_code, group_name, keyword_ko, normalized_ko,
     keyword_type, match_scope, weight, is_active, note) in rows:
    if not is_active or not category_code or category_code not in category_code_to_name:
        continue
    if keyword_type != "SPECIFIC_MENU":
        continue
    cat_keywords[category_code].append((weight or 0, keyword_ko))

CATEGORIES: dict[str, list[str]] = {}
for code, name in category_code_to_name.items():
    items = sorted(cat_keywords.get(code, []), key=lambda x: -x[0])
    seen: list[str] = []
    for _, kw in items:
        if kw not in seen:
            seen.append(kw)
        if len(seen) >= 6:
            break
    CATEGORIES[name] = seen

# ---- keyword_alias ----
# CANONICAL_REPLACEMENT(1:1 치환)만 단순 문자열 매핑으로 반영한다. MULTI_ENTITY_EXPANSION(예: "중국 국수" →
# "중식+면")은 이 서비스의 ALIASES가 1:1 치환만 지원하므로("+"를 공백으로 바꿔 "중식 면"처럼 두 토큰을 함께
# 노출하는 절충안) is_active인 것만 반영한다.
ws = wb["keyword_alias"]
rows = [r for r in ws.iter_rows(min_row=3, values_only=True) if r[0] not in (None, "alias_id")]

ALIASES: dict[str, str] = {}
for (alias_id, alias_ko, canonical_ko, entity_type, expansion_rule, is_bidirectional, is_active, note) in rows:
    if not is_active or not alias_ko or not canonical_ko:
        continue
    canonical = canonical_ko.replace("+", " ")
    ALIASES[alias_ko] = canonical

# ---- atmosphere_fallback (엑셀에 없음 — 기존 손작업 값 유지) ----
ATMOSPHERE_FALLBACK = {
    "감성": ["다이닝", "와인바", "칵테일바", "라운지바", "오마카세", "일본가정식"],
    "조용한": ["와인바", "오마카세", "다이닝", "일본가정식", "카페"],
    "데이트": ["다이닝", "와인바", "오마카세", "브런치", "칵테일바"],
    "활기찬": ["호프", "포차", "펍", "고깃집", "분식집"],
    "모임": ["고깃집", "뷔페", "호프", "샤브샤브", "한정식"],
}

# ---- review_tags (프론트 리뷰 작성 화면의 고정 태그 20개, 2026-07-22 전달받음) ----
# "대기시간길어요"/"대기시간짧아요"는 원래 긍정/부정이 서로 바뀌어 전달됐다고 확인받아(단순 오타) 여기서
# 바로잡았다 — 대기시간이 짧은 쪽이 긍정, 긴 쪽이 부정.
REVIEW_TAGS = {
    "positive": [
        "맛있어요", "재료가 신선해요", "양이 많아요", "가성비가 좋아요", "깨끗해요",
        "친절해요", "차분해요", "대기시간짧아요", "음식이빨리나와요", "가게가예뻐요",
    ],
    "negative": [
        "맛이 아쉬워요", "재료가 신선하지 않아요", "양이 적어요", "가격이 비싸요", "지저분해요",
        "불친절해요", "소란스러워요", "대기시간길어요", "음식이늦게나와요", "가게가부산스러워요",
    ],
}

master = {
    "categories": CATEGORIES,
    "aliases": ALIASES,
    "atmosphere_fallback": ATMOSPHERE_FALLBACK,
    "review_tags": REVIEW_TAGS,
}

with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(master, f, ensure_ascii=False, indent=2)

print("OK categories=%d aliases=%d" % (len(CATEGORIES), len(ALIASES)))
