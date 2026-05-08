@echo off
chcp 65001 >nul 2>&1
title Bitget Trading Agent

echo.
echo  Bitget Agent Starting...
echo  종료: Ctrl+C
echo  ─────────────────────────
echo.

call venv\Scripts\activate.bat
python main.py

if errorlevel 1 (
    echo.
    echo [ERR] 에이전트가 비정상 종료되었습니다.
    echo       logs\trading.log 를 확인하세요.
    pause
)
