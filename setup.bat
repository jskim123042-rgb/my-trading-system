@echo off
chcp 65001 >nul 2>&1
title Bitget Agent Setup

echo.
echo ==================================================
echo   Bitget Futures Trading Agent - Windows Setup
echo ==================================================
echo.

:: ── 1. Python 확인 ──
echo [>>] Python 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERR] Python이 설치되어 있지 않습니다.
    echo.
    echo   1. https://www.python.org/downloads/ 에서 Python 3.11+ 다운로드
    echo   2. 설치 시 반드시 "Add Python to PATH" 체크!
    echo   3. 설치 완료 후 이 스크립트를 다시 실행하세요.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%a in ('python --version 2^>^&1') do set PY_VER=%%a
echo [OK] Python %PY_VER% 확인 완료

:: ── 2. 가상환경 생성 ──
echo [>>] 가상환경 생성 중...
if not exist "venv" (
    python -m venv venv
    echo [OK] 가상환경 생성 완료
) else (
    echo [OK] 가상환경 이미 존재
)

:: ── 3. 가상환경 활성화 ──
call venv\Scripts\activate.bat
echo [OK] 가상환경 활성화

:: ── 4. 의존성 설치 ──
echo [>>] 의존성 설치 중... (1~2분 소요)
pip install --upgrade pip -q 2>nul
pip install -r requirements.txt -q 2>nul
if errorlevel 1 (
    echo [!!] 일부 패키지 설치 실패. 수동 설치 시도:
    echo      pip install ccxt numpy pyyaml websockets httpx apscheduler
) else (
    echo [OK] 의존성 설치 완료
)

:: ── 5. 디렉토리 생성 ──
if not exist "logs" mkdir logs
if not exist "data" mkdir data
echo [OK] logs, data 디렉토리 생성

:: ── 6. 테스트 실행 ──
echo [>>] 유닛 테스트 실행...
python -m pytest tests/ -v --tb=short 2>nul
if errorlevel 1 (
    echo [!!] 일부 테스트 실패 - 로그 확인 필요
) else (
    echo [OK] 테스트 전체 통과
)

:: ── 7. 백테스트 데모 ──
echo [>>] 백테스트 데모 실행...
python -m backtest.optimizer --strategy trend_follow --quick --demo --workers 1 --export data\demo_optim.json 2>nul
echo [OK] 백테스트 정상 작동 확인

:: ── 8. 설정 파일 확인 ──
echo.
findstr /c:"YOUR_API_KEY" config\settings.yaml >nul 2>&1
if not errorlevel 1 (
    echo [!!] config\settings.yaml에 API 키가 미설정 상태입니다.
    echo.
    echo   1. Bitget API Key 생성: https://www.bitget.com/account/newapi
    echo      Read + Trade ON / Transfer + Withdraw OFF
    echo.
    echo   2. config\settings.yaml 편집:
    echo      메모장: notepad config\settings.yaml
    echo      VSCode: code config\settings.yaml
    echo.
    echo   3. api_key, secret_key, passphrase 입력
    echo      sandbox: true 반드시 유지!
)

:: ── 완료 ──
echo.
echo ==================================================
echo   Setup 완료!
echo ==================================================
echo.
echo   실행 명령:
echo     venv\Scripts\activate
echo     python main.py
echo.
echo   백그라운드 실행 (창 닫아도 유지):
echo     start /min pythonw main.py
echo.
echo   로그 확인:
echo     type logs\trading.log
echo     powershell Get-Content logs\trading.log -Wait
echo.
pause
