------------------------------------------------------------------------
# Python 저장소 Git 브랜치 전략 및 적용 현황
------------------------------------------------------------------------
2026-07-30 기준. `soldesk-org/SoldeskProject2_Python` 저장소에 실제로 적용된 브랜치 구조와
현재 상태를 정리한 문서. (Web 저장소는 `SoldeskProject2_Web/docs/00.공통`에 별도 문서 있음)

------------------------------------------------------------------------
## 1. 브랜치 전략
------------------------------------------------------------------------

```
feature/* = 각자 만든 기능을 올리고 공유하는 곳
develop   = 모든 기능을 합치는 개발 통합본
main      = 실제 서비스에 배포하는 최종본
```

```
develop
  ├─ feature/py-data-preprocessing 생성
  │    └─ 데이터 전처리 기능 개발 및 push
  │         └─ PR: feature/py-data-preprocessing → develop
  │
  ├─ feature/py-fastapi-server 생성
  │    └─ FastAPI 서버 개발 및 push
  │         └─ PR: feature/py-fastapi-server → develop
  │
  ├─ feature/py-food-mbti 생성
  │    └─ 음식 MBTI 기능 개발 및 push
  │         └─ PR: feature/py-food-mbti → develop
  │
  ├─ feature/py-nlp-keywords 생성
  │    └─ NLP 키워드 추출 기능 개발 및 push
  │         └─ PR: feature/py-nlp-keywords → develop
  │
  └─ feature/py-similarity-model 생성
       └─ 유사도 모델 개발 및 push
            └─ PR: feature/py-similarity-model → develop

모든 기능을 develop에서 통합 및 테스트
  └─ PR: develop → main
       └─ main 병합
            └─ VM에 Python 서비스 자동 배포
```

------------------------------------------------------------------------
## 2. 현재 적용 현황 (2026-07-30)
------------------------------------------------------------------------

| 브랜치 | 상태 | 내용 |
|---|---|---|
| `feature/py-data-preprocessing` | ✅ 완료, develop 병합됨 | `build_keyword_master.py`, `keyword_master.json`, `web_menu_keyword_db.xlsx` |
| `feature/py-nlp-keywords` | ✅ 완료, develop 병합됨 | `keyword_api.py`(자연어 검색/리뷰 키워드 분석 FastAPI 서버) |
| `feature/py-fastapi-server` | ✅ 완료, develop 병합됨 | `business_auth.py`, `config.py`, `main.py`, `receipt_ocr.py` (사업자 인증/영수증 OCR 서버) |
| `feature/py-similarity-model` | ✅ 완료, develop 병합됨 | `recommendation_api.py`(AI 추천 유사도 모델, 기존 fastapi-server에서 분리) |
| `feature/py-food-mbti` | ⏳ 브랜치만 생성, 코드 없음 | 담당자 개발 대기 중 |
| `develop → main` | ✅ merge 완료, push 됨 | 위 4개 기능 통합, `py_compile` 스모크 테스트 통과 |
| main → VM 자동 배포 | ⚠️ 미연결 | 저장소에 Jenkinsfile 없음 — CI/CD 파이프라인 구축 필요 (Web 저장소의 `CI-CD-Jenkins-구축-가이드.md` 참고 가능) |

------------------------------------------------------------------------
## 3. 참고
------------------------------------------------------------------------
- `review-filter` 폴더는 Python이 아닌 Java(웹 백엔드)로 이관 예정이라 이 저장소에서 제외함.
- `keyword-extraction/run_recommendation_api.cmd`에 카카오 API 키가 평문으로 있어 커밋하지 않음 → `.env`로 이전 필요.
- `keyword-extraction/keyword_extraction.py` + 구버전 `main.py`는 `keyword_api.py`(신버전)로 대체된 것으로 보여 이번 반영에서 제외함. 불필요하면 삭제 검토.
- GitHub CLI(`gh`)가 로컬에 없어 실제 PR 화면 없이 `--no-ff` merge commit으로 동일한 이력만 남기고 직접 push함.
