"""
모의 전략 (Paper Trading) - 실거래 없이 로그만 쌓음

Strategy A: RSI + 볼린저밴드 + 반전캔들 (역추세)
  - RSI 과매도(<30) → 롱 후보, 과매수(>70) → 숏 후보
  - 가격이 BB 하단 터치 (롱) / 상단 터치 (숏)
  - 반전캔들 확인: 불리시 엔걸핑 또는 해머 (롱) / 베어리시 엔걸핑 또는 슈팅스타 (숏)
  - SL: ATR×1.5, TP: BB 중심선 또는 2R

Strategy B: RSI 다이버전스 + MACD (추세 반전)
  - RSI 강세 다이버전스 (가격 저점↓ + RSI 저점↑) → 롱
  - RSI 약세 다이버전스 (가격 고점↑ + RSI 고점↓) → 숏
  - MACD 히스토그램 크로스 확인
  - 엔걸핑 캔들 확인
  - SL: ATR×1.5, TP: 2.5R
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from strategy.base import rsi, macd, bollinger_bands, atr, ema


@dataclass
class PaperSignal:
    strategy: str          # "A" | "B"
    side: str              # "long" | "short" | "neutral"
    entry_price: float
    stop_loss: float
    take_profit: float
    rsi_val: float
    adx_val: float
    bb_pct_b: float
    reason: str


# ─────────────────────────────────────────────────────
# 공용 유틸
# ─────────────────────────────────────────────────────

def _is_bullish_engulfing(opens: np.ndarray, closes: np.ndarray, idx: int) -> bool:
    """불리시 엔걸핑: 직전 봉 음봉, 현재 봉 양봉 + 현재 봉 몸통이 직전 봉 완전 포함"""
    if idx < 1:
        return False
    prev_o, prev_c = opens[idx - 1], closes[idx - 1]
    curr_o, curr_c = opens[idx], closes[idx]
    prev_bear = prev_c < prev_o
    curr_bull = curr_c > curr_o
    engulf = curr_o <= prev_c and curr_c >= prev_o
    return prev_bear and curr_bull and engulf


def _is_bearish_engulfing(opens: np.ndarray, closes: np.ndarray, idx: int) -> bool:
    """베어리시 엔걸핑: 직전 봉 양봉, 현재 봉 음봉 + 현재 봉 몸통이 직전 봉 완전 포함"""
    if idx < 1:
        return False
    prev_o, prev_c = opens[idx - 1], closes[idx - 1]
    curr_o, curr_c = opens[idx], closes[idx]
    prev_bull = prev_c > prev_o
    curr_bear = curr_c < curr_o
    engulf = curr_o >= prev_c and curr_c <= prev_o
    return prev_bull and curr_bear and engulf


def _is_hammer(opens: np.ndarray, closes: np.ndarray,
               highs: np.ndarray, lows: np.ndarray, idx: int) -> bool:
    """해머: 아래꼬리가 몸통의 2배 이상, 위꼬리 작음"""
    o, c, h, l = opens[idx], closes[idx], highs[idx], lows[idx]
    body = abs(c - o)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if body == 0:
        return False
    return lower_wick >= body * 2.0 and upper_wick <= body * 0.5


def _is_shooting_star(opens: np.ndarray, closes: np.ndarray,
                      highs: np.ndarray, lows: np.ndarray, idx: int) -> bool:
    """슈팅스타: 위꼬리가 몸통의 2배 이상, 아래꼬리 작음"""
    o, c, h, l = opens[idx], closes[idx], highs[idx], lows[idx]
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if body == 0:
        return False
    return upper_wick >= body * 2.0 and lower_wick <= body * 0.5


def _calc_adx_simple(highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, period: int = 14) -> float:
    """간단한 ADX 계산 (마지막 값만 반환)"""
    from strategy.base import adx as _adx_fn
    arr = _adx_fn(highs, lows, closes, period)
    val = arr[-1]
    return float(val) if not np.isnan(val) else 0.0


# ─────────────────────────────────────────────────────
# Strategy A: RSI + BB + 반전캔들 (역추세)
# ─────────────────────────────────────────────────────

class PaperStrategyA:
    """
    역추세 전략
    진입: RSI 과매도/과매수 + BB 하단/상단 터치 + 반전캔들
    SL: ATR×1.5
    TP: BB 중심선 (최소 1R 이상, 최대 2R)
    """

    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    BB_PERIOD = 20
    BB_STD = 2.0
    ATR_PERIOD = 14
    SL_ATR_MULT = 1.5

    def analyze(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        current_price: float,
    ) -> PaperSignal:
        neutral = PaperSignal(
            strategy="A", side="neutral",
            entry_price=current_price, stop_loss=0.0, take_profit=0.0,
            rsi_val=0.0, adx_val=0.0, bb_pct_b=0.5, reason="neutral",
        )

        min_len = max(self.RSI_PERIOD + 5, self.BB_PERIOD + 5, self.ATR_PERIOD + 5)
        if len(closes) < min_len:
            return neutral

        rsi_arr = rsi(closes, self.RSI_PERIOD)
        rsi_val = float(rsi_arr[-2])  # 확정된 직전 봉 기준
        if np.isnan(rsi_val):
            return neutral

        bb_upper, bb_mid, bb_lower = bollinger_bands(closes, self.BB_PERIOD, self.BB_STD)
        atr_arr = atr(highs, lows, closes, self.ATR_PERIOD)
        atr_val = float(atr_arr[-2])
        if atr_val <= 0 or np.isnan(atr_val):
            return neutral

        # BB %B (현재 가격 기준)
        bb_range = float(bb_upper[-1]) - float(bb_lower[-1])
        bb_pct_b = (current_price - float(bb_lower[-1])) / bb_range if bb_range > 0 else 0.5

        adx_val = _calc_adx_simple(highs, lows, closes)

        # 반전캔들 체크 (직전 확정봉 idx = -2)
        idx = len(closes) - 2
        bull_candle = (
            _is_bullish_engulfing(opens, closes, idx) or
            _is_hammer(opens, closes, highs, lows, idx)
        )
        bear_candle = (
            _is_bearish_engulfing(opens, closes, idx) or
            _is_shooting_star(opens, closes, highs, lows, idx)
        )

        sl_dist = atr_val * self.SL_ATR_MULT

        # ─── 롱 조건 ───
        if (rsi_val < self.RSI_OVERSOLD
                and closes[-2] <= bb_lower[-2]
                and bull_candle):
            sl = round(current_price - sl_dist, 2)
            # TP: BB 중심선, 최소 1R, 최대 2R
            tp_target = float(bb_mid[-1])
            tp_1r = current_price + sl_dist
            tp_2r = current_price + sl_dist * 2
            tp = round(max(tp_1r, min(tp_target, tp_2r)), 2)
            return PaperSignal(
                strategy="A", side="long",
                entry_price=current_price, stop_loss=sl, take_profit=tp,
                rsi_val=round(rsi_val, 2), adx_val=round(adx_val, 2),
                bb_pct_b=round(bb_pct_b, 4),
                reason=f"RSI={rsi_val:.1f}<30 BB하단터치 반전캔들",
            )

        # ─── 숏 조건 ───
        if (rsi_val > self.RSI_OVERBOUGHT
                and closes[-2] >= bb_upper[-2]
                and bear_candle):
            sl = round(current_price + sl_dist, 2)
            tp_target = float(bb_mid[-1])
            tp_1r = current_price - sl_dist
            tp_2r = current_price - sl_dist * 2
            tp = round(min(tp_1r, max(tp_target, tp_2r)), 2)
            return PaperSignal(
                strategy="A", side="short",
                entry_price=current_price, stop_loss=sl, take_profit=tp,
                rsi_val=round(rsi_val, 2), adx_val=round(adx_val, 2),
                bb_pct_b=round(bb_pct_b, 4),
                reason=f"RSI={rsi_val:.1f}>70 BB상단터치 반전캔들",
            )

        return neutral


# ─────────────────────────────────────────────────────
# Strategy B: RSI 다이버전스 + MACD (추세 반전)
# ─────────────────────────────────────────────────────

class PaperStrategyB:
    """
    추세반전 전략
    진입: RSI 다이버전스 + MACD 크로스 + 엔걸핑 캔들
    SL: ATR×1.5
    TP: 2.5R

    Ross Cameron 원칙:
    - RSI 50선 근처(40~60) 제외 — 변동성 부족 구간
    - 롱: RSI < 45 (과매도 영역)
    - 숏: RSI > 55 (과매수 영역)
    """

    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14
    SL_ATR_MULT = 1.5
    TP_RR = 2.5
    # 다이버전스 탐색 룩백
    DIV_LOOKBACK = 20
    # RSI 50선 필터: 이 범위 밖에서만 진입
    RSI_LONG_MAX = 45   # 롱 진입 시 RSI 상한
    RSI_SHORT_MIN = 55  # 숏 진입 시 RSI 하한

    def _find_divergence(
        self,
        closes: np.ndarray,
        rsi_arr: np.ndarray,
        lookback: int,
    ) -> tuple[bool, bool]:
        """
        강세/약세 다이버전스 탐지
        강세: 가격 저점↓ + RSI 저점↑ (불리시 다이버전스)
        약세: 가격 고점↑ + RSI 고점↓ (베어리시 다이버전스)
        lookback 범위 내에서 직전 확정봉(-2) 기준
        """
        end = len(closes) - 2  # 현재 진행 봉 제외
        start = max(1, end - lookback)

        if end < 3:
            return False, False

        cur_close = closes[end]
        cur_rsi = rsi_arr[end]
        if np.isnan(cur_rsi):
            return False, False

        bull_div = False
        bear_div = False

        for i in range(start, end - 1):
            if np.isnan(rsi_arr[i]):
                continue
            # 불리시 다이버전스: 가격 저점↓ + RSI 저점↑
            if cur_close < closes[i] and cur_rsi > rsi_arr[i]:
                bull_div = True
            # 약세 다이버전스: 가격 고점↑ + RSI 고점↓
            if cur_close > closes[i] and cur_rsi < rsi_arr[i]:
                bear_div = True

        return bull_div, bear_div

    def analyze(
        self,
        opens: np.ndarray,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        current_price: float,
    ) -> PaperSignal:
        neutral = PaperSignal(
            strategy="B", side="neutral",
            entry_price=current_price, stop_loss=0.0, take_profit=0.0,
            rsi_val=0.0, adx_val=0.0, bb_pct_b=0.5, reason="neutral",
        )

        min_len = self.MACD_SLOW + self.MACD_SIGNAL + 5
        if len(closes) < min_len:
            return neutral

        rsi_arr = rsi(closes, self.RSI_PERIOD)
        rsi_val = float(rsi_arr[-2])
        if np.isnan(rsi_val):
            return neutral

        macd_line, signal_line, macd_hist = macd(
            closes, self.MACD_FAST, self.MACD_SLOW, self.MACD_SIGNAL
        )
        # MACD 크로스: 직전 봉(-2)에서 방향 확인
        hist_cur = float(macd_hist[-2])
        hist_prev = float(macd_hist[-3]) if len(macd_hist) >= 3 else 0.0

        atr_arr = atr(highs, lows, closes, self.ATR_PERIOD)
        atr_val = float(atr_arr[-2])
        if atr_val <= 0 or np.isnan(atr_val):
            return neutral

        # BB %B
        bb_upper, _, bb_lower = bollinger_bands(closes, 20, 2.0)
        bb_range = float(bb_upper[-1]) - float(bb_lower[-1])
        bb_pct_b = (current_price - float(bb_lower[-1])) / bb_range if bb_range > 0 else 0.5

        adx_val = _calc_adx_simple(highs, lows, closes)

        # 다이버전스 탐지
        bull_div, bear_div = self._find_divergence(closes, rsi_arr, self.DIV_LOOKBACK)

        # 엔걸핑 캔들
        idx = len(closes) - 2
        bull_engulf = _is_bullish_engulfing(opens, closes, idx)
        bear_engulf = _is_bearish_engulfing(opens, closes, idx)

        # MACD 불리시 크로스: 히스토그램이 음→양
        macd_bull_cross = hist_prev < 0 and hist_cur > 0
        # MACD 베어리시 크로스: 히스토그램이 양→음
        macd_bear_cross = hist_prev > 0 and hist_cur < 0

        sl_dist = atr_val * self.SL_ATR_MULT

        # ─── 롱 조건 (RSI 과매도 영역 < 45, 50선 근처 제외) ───
        if bull_div and macd_bull_cross and bull_engulf and rsi_val < self.RSI_LONG_MAX:
            sl = round(current_price - sl_dist, 2)
            tp = round(current_price + sl_dist * self.TP_RR, 2)
            return PaperSignal(
                strategy="B", side="long",
                entry_price=current_price, stop_loss=sl, take_profit=tp,
                rsi_val=round(rsi_val, 2), adx_val=round(adx_val, 2),
                bb_pct_b=round(bb_pct_b, 4),
                reason=f"강세다이버전스+MACD크로스+엔걸핑 RSI={rsi_val:.1f}",
            )

        # ─── 숏 조건 (RSI 과매수 영역 > 55, 50선 근처 제외) ───
        if bear_div and macd_bear_cross and bear_engulf and rsi_val > self.RSI_SHORT_MIN:
            sl = round(current_price + sl_dist, 2)
            tp = round(current_price - sl_dist * self.TP_RR, 2)
            return PaperSignal(
                strategy="B", side="short",
                entry_price=current_price, stop_loss=sl, take_profit=tp,
                rsi_val=round(rsi_val, 2), adx_val=round(adx_val, 2),
                bb_pct_b=round(bb_pct_b, 4),
                reason=f"약세다이버전스+MACD크로스+엔걸핑 RSI={rsi_val:.1f}",
            )

        return neutral
