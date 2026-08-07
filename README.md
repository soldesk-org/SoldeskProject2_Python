# SoldeskProject2_Python

잇티웨이(Eatty way)의 Python 서버 모음. Spring 백엔드가 이 서버들을 프록시해서 호출한다.

## 구성

| 폴더 | 역할 | 실행 위치 | 외부 주소 |
|---|---|---|---|
| `receipt-biz-verify` | 영수증 OCR + 사업자등록 진위확인 | Azure VM | `https://api.eattyway.com` |
| `keyword-extraction` | AI 맛집 추천 (자연어 → 키워드 → 카카오 검색) | 로컬 PC (GPU 필요) | `https://ai.eattyway.com` |

AI 추천만 GPU가 필요해서 VM이 아니라 개발자 로컬 PC에서 돌리고, Cloudflare Tunnel로
외부에 노출한다. 자세한 배포 구성은 Web 저장소의
`docs/00.공통/인프라/Python-API-서버-VM-배포-2026-08-07.md` 참고.

## 실행 방법

두 서비스 모두 같은 방식이다.

```bash
cd <폴더>
python -m venv venv

# Windows
venv\Scripts\pip install -r requirements.txt
# Linux/macOS
./venv/bin/pip install -r requirements.txt

cp .env.example .env      # 실제 키 값을 채운다
```

`receipt-biz-verify`는 사업자 진위확인에 playwright를 쓰므로 브라우저도 설치해야 한다:

```bash
playwright install chromium
playwright install-deps chromium   # Linux만
```

실행:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000                 # receipt-biz-verify
python -m uvicorn recommendation_api:app --host 127.0.0.1 --port 8000   # keyword-extraction
```

두 서비스의 기본 포트가 8000으로 같으니, 한 PC에서 동시에 띄우려면 포트를 나눠야 한다.

## 주의

- `.env`에는 실제 API 키가 들어가므로 **절대 커밋하지 않는다**(`.gitignore`에 등록됨).
  각 폴더의 `.env.example`을 복사해서 쓴다.
- `venv/`, `__pycache__/`, `*.log`도 커밋 대상이 아니다.
