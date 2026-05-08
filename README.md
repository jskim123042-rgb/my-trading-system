# Bitget BTC/USDT 선물 자동매매 에이전트

Grid Search 3,456조합 + Walk-Forward 검증 완료된 자동매매 시스템.

**전략**: EMA 5/26 + MACD 8/26/7 + RSI 필터 + ATR 기반 칼손절
**검증 결과**: IS +11.4% / OOS +14.2% / PF 1.28~1.93 / MDD -10.8%
**레버리지**: 기본 8x (Safety Margin 12.6x)

---

## 1단계: 환경 준비 (5분)

### 로컬 실행

```bash
# 프로젝트 압축 해제
unzip bitget-agent.zip
cd bitget-agent

# 원클릭 셋업 (Python 확인 → venv → 의존성 → 테스트)
chmod +x setup.sh
./setup.sh
```

### 수동 설치 (setup.sh 사용 불가 시)

```bash
cd bitget-agent

# Python 3.10+ 확인
python3 --version

# 가상환경 생성 + 활성화
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 디렉토리 생성
mkdir -p logs data
```

---

## 2단계: Bitget API 키 발급 (3분)

1. https://www.bitget.com/account/newapi 접속
2. API 이름: `trading-agent`
3. 권한 설정:
   - Read: ON
   - Trade: ON
   - Transfer: OFF (절대 금지)
   - Withdraw: OFF (절대 금지)
4. IP 화이트리스트: 서버 IP 등록 (필수)
5. 생성 후 API Key, Secret Key, Passphrase 저장

```bash
# 설정 파일 편집
nano config/settings.yaml
```

아래 3줄만 수정:

```yaml
exchange:
  api_key: "bg_xxxxxxxxxxxxx"
  secret_key: "xxxxxxxxxxxxxxxxxxxxx"
  passphrase: "your_passphrase_here"
  sandbox: true    # ← 반드시 true 유지!
```

---

## 3단계: 데모 모드 실행 (2주간)

```bash
source venv/bin/activate

# 데모 모드 실행 (sandbox: true)
python3 main.py
```

정상 실행 시 출력:

```
✅ Bitget 클라이언트 초기화 완료 | BTC/USDT:USDT | 8x | sandbox=True
💰 계좌 잔고: 10000.00 USDT (가용: 10000.00)
📊 15m 캔들 200개 로드
🔌 WebSocket 연결 완료 (public)
```

### 백그라운드 실행 (터미널 종료 후에도 유지)

```bash
# nohup 방식
nohup python3 main.py > logs/agent.log 2>&1 &
echo $! > agent.pid

# 로그 확인
tail -f logs/agent.log

# 종료
kill $(cat agent.pid)
```

### systemd 서비스 등록 (리눅스 서버)

```bash
sudo tee /etc/systemd/system/bitget-agent.service << 'EOF'
[Unit]
Description=Bitget Trading Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bitget-agent
ExecStart=/home/ubuntu/bitget-agent/venv/bin/python3 main.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bitget-agent
sudo systemctl start bitget-agent

# 상태 확인
sudo systemctl status bitget-agent

# 로그
sudo journalctl -u bitget-agent -f
```

---

## 4단계: Docker 배포 (권장)

```bash
# 빌드 + 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f agent

# 재시작
docker-compose restart agent

# 중지
docker-compose down
```

---

## 5단계: 텔레그램 알림 설정 (선택)

1. 텔레그램에서 @BotFather 검색 → `/newbot` → 봇 생성 → 토큰 복사
2. 텔레그램에서 @userinfobot 검색 → 아무 메시지 전송 → chat_id 확인
3. settings.yaml 수정:

```yaml
telegram:
  enabled: true
  bot_token: "123456:ABC-DEF1234..."
  chat_id: "987654321"
```

알림 내용:
- 포지션 진입/청산 (진입가, SL, TP, 수익률)
- 손절 발동
- 에러 발생
- 일간 요약 (거래 수, 승률, PF, 수익률)

---

## 6단계: 실거래 전환 체크리스트

데모 2주 이상 실행 후, 아래 조건을 모두 충족해야 실거래 전환:

```
□ 데모 2주간 총 수익률 양수
□ Profit Factor ≥ 1.2
□ 최대 드로다운 < 20%
□ 일간 손실 -5% 트리거가 정상 작동 확인
□ 텔레그램 알림 정상 수신 확인
□ 3연패 쿨다운이 정상 작동 확인
□ WebSocket 재연결이 자동으로 되는지 확인
```

전환:

```yaml
# config/settings.yaml
exchange:
  sandbox: false    # ← false로 변경
```

투입 금액: 총 투자 가능 금액의 5% 이하로 시작.

---

## 프로젝트 구조

```
bitget-agent/
├── config/
│   ├── __init__.py          # Config 로더
│   └── settings.yaml        # 최적화 완료 설정
├── core/
│   ├── client.py            # Bitget ccxt 래퍼
│   ├── data_feed.py         # WebSocket + REST 데이터 수집
│   └── order_manager.py     # 주문 실행 / TP / SL / 트레일링
├── strategy/
│   ├── base.py              # 인디케이터 (EMA/RSI/MACD/BB/ATR)
│   ├── trend_follow.py      # EMA 5/26 + MACD 8/26/7 전략
│   ├── breakout.py          # 볼린저 브레이크아웃 보조
│   ├── scalping.py          # 오더북 임밸런스 (향후)
│   └── aggregator.py        # 다중 전략 가중투표
├── risk/
│   └── risk_manager.py      # 포지션사이징 + 드로다운가드 + 레버리지
├── monitor/
│   └── telegram_bot.py      # 텔레그램 알림
├── backtest/
│   ├── engine.py            # 백테스트 엔진
│   ├── fast_backtest.py     # 고속 벡터화 백테스터
│   ├── param_grids.py       # 파라미터 그리드 정의
│   └── optimizer.py         # 그리드 서치 최적화
├── tests/
│   └── test_core.py         # 유닛 테스트
├── main.py                  # 에이전트 엔트리포인트
├── setup.sh                 # 원클릭 셋업
├── requirements.txt         # Python 의존성
├── Dockerfile               # Docker 이미지
└── docker-compose.yaml      # Docker 실행 설정
```

---

## 매매 규칙 요약

| 항목 | 값 |
|------|-----|
| 전략 | EMA 5/26 크로스 + MACD 8/26/7 히스토그램 + RSI 필터 |
| 레버리지 | 기본 8x (범위 5~12x, ATR 변동성 연동) |
| 손절 | ATR(14) × 1.5 (진입 즉시 서버 사이드 SL) |
| 익절 | 손절폭 × 3 (R:R = 1:3) |
| 트레일링 | 1R 수익 시 전환, 고점 -1.5% 청산 |
| 포지션 사이즈 | 계좌의 2% 리스크 / 최대 30% |
| 일간 한도 | 최대 10회 거래 / 최대 -5% 손실 |
| 쿨다운 | 3연패 시 60분 |

---

## 유용한 명령어

```bash
# 최적화 (데모 데이터)
python3 -m backtest.optimizer --strategy trend_follow --demo --walk-forward 0.7

# 최적화 (실제 API 데이터)
python3 -m backtest.optimizer --strategy trend_follow --days 30 --walk-forward 0.7

# 백테스트
python3 -m backtest.engine --days 30 --leverage 8

# 테스트
python3 -m pytest tests/ -v
```

---

## ⚠️ 경고

- **투자 원금 손실 가능.** 선물거래는 원금 초과 손실 위험이 있습니다.
- **백테스트/최적화 결과가 미래 수익을 보장하지 않습니다.**
- **감당 가능한 금액만 투입하세요.**
- API 키는 출금 권한 없이, IP 화이트리스트와 함께 생성하세요.
- 데모 모드에서 최소 2주 검증 후 실거래 전환하세요.
- 이 소프트웨어 사용으로 인한 손실에 대해 책임지지 않습니다.
