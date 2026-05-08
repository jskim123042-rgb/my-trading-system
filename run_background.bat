@echo off
chcp 65001 >nul 2>&1
title Bitget Agent [Background]

:: 가상환경 활성화
call venv\Scripts\activate.bat

echo [%date% %time%] Agent starting... >> logs\restart.log

:loop
echo [%date% %time%] Starting agent... >> logs\restart.log
python main.py >> logs\agent_stdout.log 2>&1

echo [%date% %time%] Agent stopped (exit code: %errorlevel%). Restarting in 10s... >> logs\restart.log
timeout /t 10 /nobreak >nul
goto loop
