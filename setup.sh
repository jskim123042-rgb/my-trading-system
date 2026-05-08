#!/usr/bin/env bash
set -e
echo ""
echo "=========================================="
echo "  Bitget Agent — Linux/Mac Setup"
echo "=========================================="
echo ""

python3 --version || { echo "Python3 필요. sudo apt install python3 python3-pip python3-venv"; exit 1; }

[ ! -d "venv" ] && python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
mkdir -p logs data

echo "[OK] Setup 완료"
echo ""
echo "  실행: source venv/bin/activate && python3 main.py"
echo "  백그라운드: nohup python3 main.py > logs/agent.log 2>&1 &"
echo ""
