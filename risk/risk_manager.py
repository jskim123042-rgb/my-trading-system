"""리스크 관리 모듈 - 포지션 사이징, 드로다운 가드, 동적 레버리지"""
import logging
import time
from dataclasses import dataclass, field

from config import Config

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    포지션 사이징 엔진
    - 고정 비율법: 계좌의 N%를 리스크로 사용
    - 켈리 기준: 승률/손익비 기반 최적 비율 (반켈리 적용)
    """

    def __init__(self, config: Config):
        self.config = config
        self.risk_per_trade = config.risk.get("risk_per_trade_pct", 0.02)
        self.max_position = config.risk.get("max_position_pct", 0.3)

    def calculate_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        leverage: int,
        win_rate: float = 0.0,
        avg_rr: float = 0.0,
        total_trades: int = 0,
    ) -> float:
        """
        진입 수량 계산

        Returns: BTC 수량 (소수점)
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0

        # 손절폭 (%)
        sl_pct = abs(entry_price - stop_loss) / entry_price
        if sl_pct <= 0:
            return 0.0

        # SL이 너무 좁으면 최소값 강제 (0.2% 미만은 슬리피지/갭 위험)
        sl_pct = max(sl_pct, 0.002)

        # 레버리지 반영 포지션 사이징:
        # 명목가치 = (잔고 × risk% × 레버리지) / SL폭
        # 레버리지가 높을수록 같은 증거금으로 더 큰 포지션
        # 예) 잔고$1000, risk2%, SL1%, 레버리지8x → 명목 = $1000×0.02×8/0.01 = $16,000
        risk_amount = balance * self.risk_per_trade
        position_value = risk_amount * leverage / sl_pct

        # 켈리 기준: 30회 이상 트레이드 히스토리가 있을 때만 적용
        if total_trades >= 30 and win_rate > 0 and avg_rr > 0:
            kelly = (win_rate * avg_rr - (1 - win_rate)) / avg_rr
            half_kelly = max(0.0, kelly * 0.5)
            if half_kelly > 0:
                kelly_value = balance * half_kelly * leverage
                kelly_value = max(kelly_value, position_value * 0.5)
                position_value = min(position_value, kelly_value)

        # 증거금 캡: 잔고의 max_position_pct 이하
        # 명목가치 캡 = 증거금 캡 × 레버리지
        max_margin = balance * self.max_position
        max_value = max_margin * leverage
        position_value = min(position_value, max_value)

        # BTC 수량으로 변환 (OKX 최소단위 0.01 BTC)
        CONTRACT_SIZE = 0.01  # OKX BTC-USDT-SWAP 1계약 = 0.01 BTC
        amount_raw = position_value / entry_price
        # 0.01 단위로 올림 (목표 리스크를 충족하도록 올림 처리)
        import math
        amount = math.ceil(amount_raw / CONTRACT_SIZE) * CONTRACT_SIZE
        amount = round(amount, 4)

        # 최소 1계약 보장
        amount = max(amount, CONTRACT_SIZE)

        # 올림 후 증거금이 캡 초과하면 다시 내림으로
        used_margin = amount * entry_price / leverage
        if used_margin > max_margin:
            amount = math.floor(amount_raw / CONTRACT_SIZE) * CONTRACT_SIZE
            amount = max(round(amount, 4), CONTRACT_SIZE)
            used_margin = amount * entry_price / leverage

        nominal = amount * entry_price
        actual_risk = nominal * sl_pct   # SL 터질 때 실제 손실 USDT
        logger.info(
            f"📐 포지션 사이징 | 잔고={balance:.2f} USDT | "
            f"레버리지={leverage}x | SL폭={sl_pct:.2%} | "
            f"수량={amount} BTC | 명목={nominal:.2f} USDT | "
            f"증거금={used_margin:.2f} USDT ({used_margin/balance:.1%}) | "
            f"손절손실={actual_risk:.2f} USDT ({actual_risk/balance:.1%}) | "
            f"증거금캡={max_margin:.0f} USDT ({self.max_position:.0%})"
        )
        return amount


@dataclass
class DrawdownState:
    """일간 드로다운 상태 추적"""
    day_start_balance: float = 0.0
    day_start_time: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: float = 0.0
    is_paused: bool = False
    pause_reason: str = ""


class DrawdownGuard:
    """
    드로다운 보호 시스템
    - 일간 최대 손실 제한
    - 일간 최대 거래 횟수 제한
    - 연패 시 쿨다운
    """

    def __init__(self, config: Config):
        self.config = config
        self.state = DrawdownState()
        
        self.max_daily_loss = config.risk.get("daily_max_loss_pct", 0.05)
        self.max_daily_trades = config.risk.get("daily_max_trades", 10)
        self.consecutive_loss_limit = config.risk.get("consecutive_loss_pause", 3)
        self.cooldown_minutes = config.risk.get("cooldown_minutes", 60)

    def reset_daily(self, current_balance: float):
        """일간 상태 리셋 (매일 00:00 UTC)"""
        self.state.day_start_balance = current_balance
        self.state.day_start_time = time.time()
        self.state.daily_trades = 0
        self.state.daily_pnl = 0.0
        self.state.is_paused = False
        self.state.pause_reason = ""
        logger.info(f"📅 일간 리셋 | 시작 잔고={current_balance:.2f} USDT")

    def can_trade(self, current_balance: float) -> tuple[bool, str]:
        """트레이딩 가능 여부 판단"""
        now = time.time()

        # 1. 쿨다운 중인지 확인
        if now < self.state.cooldown_until:
            remaining = (self.state.cooldown_until - now) / 60
            return False, f"쿨다운 중 (잔여 {remaining:.0f}분)"

        # 2. 일간 최대 손실 체크
        if self.state.day_start_balance > 0:
            daily_loss_pct = (
                (current_balance - self.state.day_start_balance)
                / self.state.day_start_balance
            )
            if daily_loss_pct < -self.max_daily_loss:
                self.state.is_paused = True
                self.state.pause_reason = f"일간 손실 {daily_loss_pct:.2%} (한도 {-self.max_daily_loss:.1%})"
                return False, self.state.pause_reason

        # 3. 일간 최대 거래 횟수
        if self.state.daily_trades >= self.max_daily_trades:
            return False, f"일간 거래 한도 도달 ({self.state.daily_trades}/{self.max_daily_trades})"

        return True, "OK"

    def record_trade(self, pnl: float):
        """트레이드 결과 기록"""
        self.state.daily_trades += 1
        self.state.daily_pnl += pnl

        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.consecutive_loss_limit:
                self.state.cooldown_until = time.time() + self.cooldown_minutes * 60
                logger.warning(
                    f"⏸️ {self.state.consecutive_losses}연패 → "
                    f"{self.cooldown_minutes}분 쿨다운 발동"
                )
                self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses = 0

    def get_status(self) -> dict:
        return {
            "daily_trades": self.state.daily_trades,
            "daily_pnl": self.state.daily_pnl,
            "consecutive_losses": self.state.consecutive_losses,
            "is_paused": self.state.is_paused,
            "pause_reason": self.state.pause_reason,
        }


class LeverageController:
    """
    동적 레버리지 조절
    - 변동성(ATR/가격) 높을 때 → 레버리지 감소
    - 연승 시 점진적 증가
    - 연패 시 즉시 최소로
    """

    def __init__(self, config: Config):
        self.config = config
        self.current = config.leverage["default"]
        self.min_lev = config.leverage["min"]
        self.max_lev = config.leverage["max"]
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def adjust_for_volatility(self, atr_value: float, price: float) -> int:
        """ATR 기반 변동성 조절"""
        if not self.config.leverage.get("volatility_adjust", True):
            return self.current

        volatility_pct = atr_value / price if price > 0 else 0

        # 변동성 → 레버리지 매핑
        if volatility_pct > 0.03:      # 3% 이상: 매우 높음
            target = self.min_lev
        elif volatility_pct > 0.02:    # 2~3%: 높음
            target = max(self.min_lev, self.current - 3)
        elif volatility_pct > 0.01:    # 1~2%: 보통
            target = self.config.leverage["default"]
        else:                          # 1% 미만: 낮음
            target = min(self.max_lev, self.current + 2)

        self.current = max(self.min_lev, min(target, self.max_lev))
        return self.current

    def adjust_for_streak(self, won: bool) -> int:
        """연승/연패에 따른 조절"""
        if won:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if self.consecutive_wins >= 3:
                self.current = min(self.current + 1, self.max_lev)
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if self.consecutive_losses >= 2:
                self.current = self.min_lev  # 즉시 최소로

        return self.current

    def adjust_for_signal_grade(self, div_grade: str) -> int:
        """
        시그널 등급별 레버리지 추가 조절 (ATR 조절 이후 叠加 적용)
        A급(고신뢰) → +2x / D·E급(저신뢰) → -1x
        """
        grade_boost_map: dict = self.config.leverage.get("grade_boost", {})
        boost = grade_boost_map.get(div_grade, 0)
        if boost == 0:
            return self.current

        target = self.current + boost
        prev = self.current
        self.current = max(self.min_lev, min(target, self.max_lev))
        logger.info(
            f"레버리지 등급 조절 | {div_grade}급 boost={boost:+d} | "
            f"{prev}x → {self.current}x"
        )
        return self.current

    def get_leverage(self) -> int:
        return self.current
