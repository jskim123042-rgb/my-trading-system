"""브레이크아웃 전략: 볼린저 밴드 돌파 + 피보나치 지지/저항 + 거래량 확인"""
import numpy as np
from strategy.base import (
    BaseStrategy, StrategyResult, Signal,
    bollinger_bands, atr, fibonacci_levels, sma,
)


class BreakoutStrategy(BaseStrategy):
    """
    진입 조건:
    - LONG: 종가가 볼린저 상단 돌파 + 거래량 1.5배 이상 + 피보나치 저항선 위
    - SHORT: 종가가 볼린저 하단 이탈 + 거래량 1.5배 이상 + 피보나치 지지선 아래
    
    핵심: 가짜 브레이크아웃 필터링 (거래량 확인 필수)
    """

    def __init__(self, params: dict):
        super().__init__("breakout", params)

    def analyze(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        current_price: float,
        **kwargs,
    ) -> StrategyResult:
        if len(closes) < 50:
            return StrategyResult(signal=Signal.NEUTRAL, reason="데이터 부족")

        bb_period = self.params.get("bb_period", 20)
        bb_std = self.params.get("bb_std", 2.0)
        fib_lookback = self.params.get("fib_lookback", 100)
        vol_mult = self.params.get("volume_confirm_multiplier", 1.5)

        # 볼린저 밴드
        upper, middle, lower = bollinger_bands(closes, bb_period, bb_std)
        
        # 피보나치 레벨
        lookback = min(fib_lookback, len(highs))
        recent_high = np.max(highs[-lookback:])
        recent_low = np.min(lows[-lookback:])
        fib = fibonacci_levels(recent_high, recent_low)

        # 거래량 확인
        vol_sma = sma(volumes, 20)
        vol_ratio = volumes[-1] / vol_sma[-1] if vol_sma[-1] > 0 else 0

        # ATR
        atr_val = atr(highs, lows, closes, 14)
        curr_atr = atr_val[-1]

        # 볼린저 밴드 위치
        bb_upper_break = closes[-1] > upper[-1] and closes[-2] <= upper[-2]
        bb_lower_break = closes[-1] < lower[-1] and closes[-2] >= lower[-2]
        
        # 볼린저 밴드 스퀴즈 (밴드폭 축소 후 돌파 = 강한 시그널)
        bb_width = (upper - lower) / middle
        bb_squeeze = bb_width[-1] < np.percentile(bb_width[-50:], 20)

        signal = Signal.NEUTRAL
        confidence = 0.0
        reason_parts = []

        # ── LONG 브레이크아웃 ──
        if bb_upper_break and vol_ratio >= vol_mult:
            # 피보나치 0.618 위에 있으면 강한 시그널
            if current_price > fib[0.382]:
                signal = Signal.STRONG_LONG if bb_squeeze else Signal.LONG
                confidence = 0.85 if bb_squeeze else 0.7
                reason_parts = [
                    "볼린저 상단 돌파",
                    f"거래량 {vol_ratio:.1f}x",
                    f"피보 0.382({fib[0.382]:.0f}) 위",
                ]
                if bb_squeeze:
                    reason_parts.append("스퀴즈 후 돌파")
            else:
                signal = Signal.LONG
                confidence = 0.55
                reason_parts = ["볼린저 상단 돌파", f"거래량 {vol_ratio:.1f}x"]

        # ── SHORT 브레이크아웃 ──
        elif bb_lower_break and vol_ratio >= vol_mult:
            if current_price < fib[0.618]:
                signal = Signal.STRONG_SHORT if bb_squeeze else Signal.SHORT
                confidence = 0.85 if bb_squeeze else 0.7
                reason_parts = [
                    "볼린저 하단 이탈",
                    f"거래량 {vol_ratio:.1f}x",
                    f"피보 0.618({fib[0.618]:.0f}) 아래",
                ]
                if bb_squeeze:
                    reason_parts.append("스퀴즈 후 이탈")
            else:
                signal = Signal.SHORT
                confidence = 0.55
                reason_parts = ["볼린저 하단 이탈", f"거래량 {vol_ratio:.1f}x"]

        # 손절/익절
        sl_distance = curr_atr * 1.5
        rr_ratio = kwargs.get("rr_ratio", 3.0)

        if signal.value > 0:
            stop_loss = current_price - sl_distance
            take_profit = current_price + sl_distance * rr_ratio
        elif signal.value < 0:
            stop_loss = current_price + sl_distance
            take_profit = current_price - sl_distance * rr_ratio
        else:
            stop_loss = take_profit = 0

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            reason=" | ".join(reason_parts) if reason_parts else "시그널 없음",
            extra={"bb_squeeze": bool(bb_squeeze)},
        )
