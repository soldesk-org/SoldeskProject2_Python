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
| `feature/py-data-preprocessing` | 완료, develop 병합됨 | `build_keyword_master.py`, `keyword_master.json`, `web_menu_keyword_db.xlsx` |
| `feature/py-nlp-keywords` | 완료, develop 병합됨 | `keyword_api.py`(자연어 검색/리뷰 키워드 분석 FastAPI 서버) |
| `feature/py-fastapi-server` | 완료, develop 병합됨 | `business_auth.py`, `config.py`, `main.py`, `receipt_ocr.py` (사업자 인증/영수증 OCR 서버) |
| `feature/py-similarity-model` | 완료, develop 병합됨 | `recommendation_api.py`(AI 추천 유사도 모델, 기존 fastapi-server에서 분리) |
| `feature/py-food-mbti` | 브랜치만 생성, 코드 없음 | 담당자 개발 대기 중 |
| `develop → main` | merge 완료, push 됨 | 위 4개 기능 통합, `py_compile` 스모크 테스트 통과 |
| main → VM 자동 배포 | 미연결 | 저장소에 Jenkinsfile 없음, CI/CD 파이프라인 구축 필요 (Web 저장소의 `CI-CD-Jenkins-구축-가이드.md` 참고 가능) |

------------------------------------------------------------------------
## 3. 작업 폴더와 git 폴더 분리
------------------------------------------------------------------------
처음에 실제 작업 폴더(`SoldeskProject2_Python`, keyword-extraction/receipt-biz-verify/review-filter가
있는 폴더)에 직접 `git init`을 했더니, 브랜치를 체크아웃할 때마다 git이 해당 브랜치의 추적 파일
(business_auth.py, keyword_api.py 등)을 작업 폴더 루트에 그대로 풀어놓아서 원래 폴더 안의 파일과
중복되어 보이는 문제가 있었음. 이를 해결하기 위해 아래와 같이 분리함.

- `SoldeskProject2_Python` (작업 폴더): keyword-extraction, receipt-biz-verify, review-filter 등
  코드 작업만 하는 공간. git과 무관하며 루트에 다른 파일이 섞이지 않음.
- `SoldeskProject2_Python-git` (git 전용 폴더): `soldesk-org/SoldeskProject2_Python` 저장소를
  clone해 둔 폴더. 앞으로 GitHub에 뭔가 올릴 일이 있으면 이 폴더에서만 브랜치 체크아웃/커밋/push를
  진행함.

앞으로 GitHub에 뭔가 올릴 일이 있으면 `SoldeskProject2_Python-git` 폴더에서 작업한다. 작업 폴더
(`SoldeskProject2_Python`)는 이제 코드 작업 전용 공간이다.

------------------------------------------------------------------------
## 4. 참고
------------------------------------------------------------------------
- `review-filter` 폴더는 Python이 아닌 Java(웹 백엔드)로 이관 예정이라 이 저장소에서 제외함.
- `keyword-extraction/run_recommendation_api.cmd`에 카카오 API 키가 평문으로 있어 커밋하지 않음, `.env`로 이전 필요.
- `keyword-extraction/keyword_extraction.py` + 구버전 `main.py`는 `keyword_api.py`(신버전)로 대체된 것으로 보여 이번 반영에서 제외함. 불필요하면 삭제 검토.
- GitHub CLI(`gh`)가 로컬에 없어 실제 PR 화면 없이 `--no-ff` merge commit으로 동일한 이력만 남기고 직접 push함.
- `requirements.txt`, `.ps1`, `.log`, `.cmd` 파일은 각 브랜치에서 git 추적 대상에서 제외함(팀 결정). 의존성 목록은 필요 시 이 문서나 별도 문서에 텍스트로 남기는 방식으로 관리.
