@echo off
REM AI 추천 서버 실행 (GPU 필요)
REM
REM 이 서버는 python-dotenv를 쓰지 않으므로 .env 파일을 자동으로 읽지 못한다.
REM 그래서 실행 전에 환경변수를 직접 지정해야 한다.
REM
REM 사용법: 이 파일을 run_recommendation_api.local.cmd 로 복사한 뒤
REM         KAKAO_REST_API_KEY 에 실제 값을 채워서 그 파일을 실행할 것.
REM         (.local.cmd 는 .gitignore 에 걸려 있어 저장소에 올라가지 않는다)
cd /d "%~dp0"

set KAKAO_REST_API_KEY=
set REVIEW_STATS_URL=http://127.0.0.1:8081/internal/reviews/keyword-ratios
set FOOTTRIP_MODEL_NAME=kakaocorp/kanana-nano-2.1b-instruct

if "%KAKAO_REST_API_KEY%"=="" (
  echo.
  echo  [!] KAKAO_REST_API_KEY 가 비어 있습니다.
  echo      이 파일을 복사해서 실제 키를 채운 뒤 실행하세요.
  echo.
  pause
  exit /b 1
)

venv\Scripts\python.exe -m uvicorn recommendation_api:app --host 127.0.0.1 --port 8000
