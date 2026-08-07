# 영수증 OCR + 사업자 진위확인 서버 실행
# 환경변수는 같은 폴더의 .env 에서 자동으로 읽는다(config.py 의 load_dotenv).
# .env.example 을 복사해서 실제 값을 채울 것.
Set-Location $PSScriptRoot
& venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
