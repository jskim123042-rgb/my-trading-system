"""스캘핑 전략: 오더북 임밸런스 + 체결 압력 분석"""
import numpy as np
from strategy.base import BaseStrategy, StrategyResult, Signal, atr


class ScalpingStrategy(BaseStrategy):
    """
    진입 조건:
    - 오더북 매수벽/매도벽 임밸런스 감지
    - bid_volume / (bid_volume + ask_volume) > threshold → LONG
    - ask_volume / (bid_volume + ask_volume) > threshold → SHORT
    
    초단타이므로 TP/SL 타이트하게 설정
    """

    def __init__(self, params: dict):
        super().__init__("scalping", params)

    def analyze(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        current_price: float,
        orderbook: dict = None,
        **kwargs,
    ) -> StrategyResult:
        if orderbook is None or not orderbook.get("bids") or not orderbook.get("asks"):
            return StrategyResult(signal=Signal.NEUTRAL, reason="오더북 데이터 없음")

        depth = self.params.get("orderbook_depth", 20)
        threshold = self.params.get("imbalance_threshold", 0.65)
        min_spread = self.params.get("min_spread_bps", 3)

        bids = orderbook["bids"][:depth]
        asks = orderbook["asks"][:depth]

        if not bids or not asks:
            return StrategyResult(signal=Signal.NEUTRAL, reason="호가 부족")

        # 스프레드 체크 (너무 넓으면 스캘핑 불리)
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread_bps = (best_ask - best_bid) / best_bid * 10000

        if spread_bps > min_spread * 3:
            return StrategyResult(signal=Signal.NEUTRAL, reason=f"스프레드 과다 ({spread_bps:.1f}bps)")

        # 오더북 임밸런스 계산
        bid_volume = sum(b[1] for b in bids)
        ask_volume = sum(a[1] for a in asks)
        total = bid_volume + ask_volume

        if total == 0:
            return StrategyResult(signal=Signal.NEUTRAL, reason="거래량 없음")

        bid_ratio = bid_volume / total
        ask_ratio = ask_volume / total

        # 가중 임밸런스 (가까운 호가에 더 높은 가중치)
        weights = np.exp(-np.arange(len(bids)) * 0.15)
        weighted_bid = sum(b[1] * w for b, w in zip(bids, weights))
        weighted_ask = sum(a[1] * w for a, w in zip(asks, weights[:len(asks)]))
        weighted_total = weighted_bid + weighted_ask
        weighted_bid_ratio = weighted_bid / weighted_total if weighted_total > 0 else 0.5

        # ATR 기반 TP/SL (스캘핑용 = 짧은 주기)
        if len(closes) >= 14:
            atr_val = atr(highs, lows, closes, 14)
            curr_atr = atr_val[-1]
        else:
            curr_atr = current_price * 0.003  # 폴백: 0.3%

        signal = Signal.NEUTRAL
        confidence = 0.0
        reason_parts = []

        # ── LONG: 매수벽 강함 ──
        if weighted_bid_ratio > threshold and bid_ratio > 0.55:
            signal = Signal.LONG
            confidence = min(0.8, (weighted_bid_ratio - 0.5) * 2)
            reason_parts = [
                f"매수 임밸런스 {weighted_bid_ratio:.1%}",
                f"bid/ask={bid_ratio:.1%}",
                f"스프레드={spread_bps:.1f}bps",
            ]

        # ── SHORT: 매도벽 강함 ──
        elif (1 - weighted_bid_ratio) > threshold and ask_ratio > 0.55:
            signal = Signal.SHORT
            confidence = min(0.8, ((1 - weighted_bid_ratio) - 0.5) * 2)
            reason_parts = [
                f"매도 임밸런스 {1 - weighted_bid_ratio:.1%}",
                f"ask/total={ask_ratio:.1%}",
                f"스프레드={spread_bps:.1f}bps",
            ]

        # 스캘핑 TP/SL (ATR × 0.8 / ATR × 0.4)
        sl_distance = curr_atr * 0.4
        tp_distance = curr_atr * 0.8

        if signal.value > 0:
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        elif signal.value < 0:
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance
        else:
            stop_loss = take_profit = 0

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            reason=" | ".join(reason_parts) if reason_parts else "시그널 없음",
        )
