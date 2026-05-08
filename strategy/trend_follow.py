"""멀티 타임프레임 다이버전스 전략 (15m / 1h / 4h 동시 확인)
btc_divergence_v4_optimized.py 로직 기반

진입 원칙:
  - 최소 2개 TF에서 같은 방향 다이버전스 확인 후 진입
  - 3개 TF 모두 확인 시 A급 (최강)

TF별 신뢰도:
  4h > 1h > 15m  (상위 TF 다이버전스가 더 신뢰 가능)

등급 체계:
  A급 (confidence 0.95): 4h + 1h + 15m  전체 확인
  B급 (confidence 0.82): 4h + 1h        두 상위 TF 확인
  C급 (confidence 0.72): 1h + 15m       두 하위 TF 확인
  D급 (confidence 0.55): 4h 단독
  E급 (confidence 0.60): 15m 단독       CVD확인 + EMA50방향 + strength≥2 필수
"""
import logging
import numpy as np
from strategy.base import (
    BaseStrategy, StrategyResult, Signal,
    ema, rsi, atr,
)

logger = logging.getLogger(__name__)


# ── 피벗 탐지 (v4와 동일 로직) ─────────────────────────────────────────────

def _find_pivot_lows(arr: np.ndarray, window: int = 5) -> list[int]:
    result = []
    n = len(arr)
    for i in range(window, n - window):
        left = arr[i - window:i]
        right = arr[i + 1:i + window + 1]
        if arr[i] < left.min() and arr[i] < right.min():
            result.append(i)
    return result


def _find_pivot_highs(arr: np.ndarray, window: int = 5) -> list[int]:
    result = []
    n = len(arr)
    for i in range(window, n - window):
        left = arr[i - window:i]
        right = arr[i + 1:i + window + 1]
        if arr[i] > left.max() and arr[i] > right.max():
            result.append(i)
    return result


# ── RSI 다이버전스 탐지 (TF별 파라미터 지원) ────────────────────────────────

def _detect_divergence(
    closes: np.ndarray,
    rsi_arr: np.ndarray,
    pivot_window: int = 5,
    max_lookback: int = 30,
    recency_bars: int = 10,
) -> tuple[bool, bool, int]:
    """
    RSI 다이버전스 탐지 (v4 로직)

    Returns:
        bull_signal  : 강세 다이버전스 여부
        bear_signal  : 약세 다이버전스 여부
        div_strength : 강도 0~4
    """
    n = len(closes)
    if n < pivot_window * 2 + 5:
        return False, False, 0

    pl = _find_pivot_lows(closes, pivot_window)
    ph = _find_pivot_highs(closes, pivot_window)

    bull_signal = bear_signal = False
    div_strength = 0

    # ── Bullish (저점 2개 비교) ───────────────────────────────────────────
    if len(pl) >= 2:
        curr_i, prev_i = pl[-1], pl[-2]
        bars_between = curr_i - prev_i
        bars_since   = (n - 1) - curr_i          # 마지막 피벗 이후 경과 봉

        if 3 <= bars_between <= max_lookback and bars_since <= recency_bars:
            c_p, p_p = closes[curr_i], closes[prev_i]
            c_r, p_r = rsi_arr[curr_i], rsi_arr[prev_i]

            # Regular Bullish: 가격 LL + RSI HL (과매도)
            if c_p < p_p and c_r > p_r and p_r < 40:
                bull_signal  = True
                div_strength = 2
                if c_r - p_r > 5: div_strength += 1   # RSI 격차 큼
                if c_r < 30:      div_strength += 1   # 깊은 과매도

            # Hidden Bullish: 가격 HL + RSI LL (상승추세 지속)
            elif c_p > p_p and c_r < p_r:
                bull_signal  = True
                div_strength = 1

    # ── Bearish (고점 2개 비교) ───────────────────────────────────────────
    if len(ph) >= 2:
        curr_i, prev_i = ph[-1], ph[-2]
        bars_between = curr_i - prev_i
        bars_since   = (n - 1) - curr_i

        if 3 <= bars_between <= max_lookback and bars_since <= recency_bars:
            c_p, p_p = closes[curr_i], closes[prev_i]
            c_r, p_r = rsi_arr[curr_i], rsi_arr[prev_i]

            # Regular Bearish: 가격 HH + RSI LH (과매수)
            if c_p > p_p and c_r < p_r and p_r > 60:
                bear_signal  = True
                div_strength = 2
                if p_r - c_r > 5: div_strength += 1
                if c_r > 70:      div_strength += 1

            # Hidden Bearish: 가격 LH + RSI HH (하락추세 지속)
            elif c_p < p_p and c_r > p_r:
                bear_signal  = True
                div_strength = 1

    return bull_signal, bear_signal, div_strength


# ── 의사 CVD (v4 calculate_pseudo_cvd 실시간 버전) ──────────────────────────

def _calc_pseudo_cvd(
    closes: np.ndarray,
    volumes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    window: int = 20,
) -> tuple[bool, bool]:
    n = min(len(closes), len(volumes), len(highs), len(lows))
    if n < window:
        return False, False

    c, v, h, lo = closes[-n:], volumes[-n:], highs[-n:], lows[-n:]
    rng = np.where(h - lo == 0, 1e-10, h - lo)
    buy_ratio = (c - lo) / rng
    cvd = np.cumsum(v * (2 * buy_ratio - 1))
    cvd_ma = np.mean(cvd[-window:])

    return bool(cvd[-1] > cvd_ma), bool(cvd[-1] < cvd_ma)


# ── TF별 파라미터 ─────────────────────────────────────────────────────────────
#  상위 TF일수록 피벗 윈도우 작게, 최근성 기준 여유 있게
_TF_PARAMS = {
    "15m": {"pivot_window": 5, "max_lookback": 30, "recency_bars": 10},
    "1h":  {"pivot_window": 4, "max_lookback": 25, "recency_bars":  7},
    "4h":  {"pivot_window": 3, "max_lookback": 20, "recency_bars":  5},
}


def _check_tf(
    closes, highs, lows, volumes, rsi_period, tf_key, min_bars=40
) -> tuple[bool, bool, int, bool, bool]:
    """
    단일 TF의 다이버전스 + CVD 확인
    closes[-1]은 현재 형성 중인 미완성 봉 → 피벗/RSI 계산에서 제외
    Returns: (bull, bear, div_strength, cvd_bull, cvd_bear)
    """
    if closes is None or len(closes) < min_bars:
        return False, False, 0, False, False

    # 미완성 봉(현재 봉) 제거 → 확정된 캔들만 사용
    c = closes[:-1]
    h = highs[:-1]
    lo = lows[:-1]
    v = volumes[:-1]

    if len(c) < min_bars:
        return False, False, 0, False, False

    p = _TF_PARAMS.get(tf_key, _TF_PARAMS["1h"])
    rsi_arr = rsi(c, rsi_period)
    bull, bear, strength = _detect_divergence(
        c, rsi_arr,
        pivot_window=p["pivot_window"],
        max_lookback=p["max_lookback"],
        recency_bars=p["recency_bars"],
    )
    cvd_bull, cvd_bear = _calc_pseudo_cvd(c, v, h, lo)
    return bull, bear, strength, cvd_bull, cvd_bear


# ── 전략 클래스 ───────────────────────────────────────────────────────────────

class TrendFollowStrategy(BaseStrategy):
    """
    멀티 타임프레임 다이버전스 전략

    진입 등급:
      A급: 4h + 1h + 15m 모두 같은 방향 다이버전스 → 0.95
      B급: 4h + 1h  상위 2 TF 확인                → 0.82
      C급: 1h + 15m 하위 2 TF 확인                → 0.72
      D급: 4h 단독                                 → 0.55
      E급: 15m 단독  CVD확인+EMA50방향+강도≥2      → 0.60
    """

    def __init__(self, params: dict):
        super().__init__("trend_follow", params)

    def analyze(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        current_price: float,
        **kwargs,
    ) -> StrategyResult:
        rsi_period   = self.params.get("rsi_period",   14)
        ema_trend_p  = self.params.get("ema_trend",    50)
        atr_mult     = self.params.get("atr_sl_mult",  1.5)
        min_bars_15m = self.params.get("min_bars",     60)

        if len(closes) < min_bars_15m:
            return StrategyResult(signal=Signal.NEUTRAL, reason="데이터 부족")

        # ── 거래량 필터 ──────────────────────────────────────────────────
        vol_ratio = kwargs.get("vol_ratio", 1.0)
        if vol_ratio < 0.5:
            return StrategyResult(signal=Signal.NEUTRAL, reason=f"거래량 부족 ({vol_ratio:.2f}x)")

        # ── EMA50 / ATR: 확정 봉 기준 (미완성 봉 제외) ─────────────────
        atr_val     = atr(highs[:-1], lows[:-1], closes[:-1], 14)[-1]
        trend_above = current_price > ema(closes[:-1], ema_trend_p)[-1]

        # ── 1h 상위 추세 방향 ────────────────────────────────────────────
        ema_1h_trend = kwargs.get("ema_1h_trend", "unknown")

        # ── 각 TF 다이버전스 확인 ────────────────────────────────────────
        bull_15m, bear_15m, str_15m, cvd_bull_15m, cvd_bear_15m = _check_tf(
            closes, highs, lows, volumes, rsi_period, "15m", min_bars_15m
        )

        closes_1h  = kwargs.get("closes_1h")
        highs_1h   = kwargs.get("highs_1h")
        lows_1h    = kwargs.get("lows_1h")
        volumes_1h = kwargs.get("volumes_1h")
        bull_1h, bear_1h, str_1h, cvd_bull_1h, cvd_bear_1h = _check_tf(
            closes_1h, highs_1h, lows_1h, volumes_1h, rsi_period, "1h"
        )

        closes_4h  = kwargs.get("closes_4h")
        highs_4h   = kwargs.get("highs_4h")
        lows_4h    = kwargs.get("lows_4h")
        volumes_4h = kwargs.get("volumes_4h")
        bull_4h, bear_4h, str_4h, cvd_bull_4h, cvd_bear_4h = _check_tf(
            closes_4h, highs_4h, lows_4h, volumes_4h, rsi_period, "4h"
        )

        # ── TF 확인 집계 ─────────────────────────────────────────────────
        long_tfs  = [bull_4h, bull_1h, bull_15m]   # [4h, 1h, 15m]
        short_tfs = [bear_4h, bear_1h, bear_15m]
        long_tf_count  = sum(long_tfs)
        short_tf_count = sum(short_tfs)

        # CVD: 상위 TF 우선 (4h CVD가 가장 신뢰도 높음)
        cvd_long  = cvd_bull_4h or cvd_bull_1h or cvd_bull_15m
        cvd_short = cvd_bear_4h or cvd_bear_1h or cvd_bear_15m
        max_long_str  = max(str_4h if bull_4h else 0,
                            str_1h if bull_1h else 0,
                            str_15m if bull_15m else 0)
        max_short_str = max(str_4h if bear_4h else 0,
                            str_1h if bear_1h else 0,
                            str_15m if bear_15m else 0)

        rr_ratio = kwargs.get("rr_ratio", 3.0)
        signal    = Signal.NEUTRAL
        confidence = 0.0
        reason_parts: list[str] = []

        def _tf_label(b4, b1, b15):
            parts = []
            if b4:  parts.append("4h")
            if b1:  parts.append("1h")
            if b15: parts.append("15m")
            return "+".join(parts)

        # ═══════════════ LONG ════════════════════════════════════════════
        if long_tf_count >= 2 and ema_1h_trend != "below":
            tf_str = _tf_label(bull_4h, bull_1h, bull_15m)

            if long_tf_count == 3 and cvd_long and trend_above:
                # A급: 전 TF 확인 + CVD + EMA50 위
                signal = Signal.STRONG_LONG
                confidence = 0.95
                reason_parts = [
                    f"[A급] 강세다이버전스 {tf_str}(강도{max_long_str})",
                    "CVD매수우위",
                    f"EMA{ema_trend_p}위",
                ]

            elif bull_4h and bull_1h:
                # B급: 상위 2 TF (4h+1h) ← 핵심 진입 기준
                h1_pen = 0.85 if ema_1h_trend == "unknown" else 1.0
                signal = Signal.STRONG_LONG if cvd_long else Signal.LONG
                confidence = round(0.82 * h1_pen, 3)
                reason_parts = [
                    f"[B급] 강세다이버전스 {tf_str}(강도{max_long_str})",
                    "CVD매수우위" if cvd_long else "CVD미확인",
                ]

            elif bull_1h and bull_15m:
                # C급: 하위 2 TF (1h+15m)
                h1_pen = 0.8 if ema_1h_trend == "below" else 1.0
                signal = Signal.LONG
                confidence = round(0.72 * h1_pen, 3)
                reason_parts = [
                    f"[C급] 강세다이버전스 {tf_str}(강도{max_long_str})",
                    "CVD매수우위" if cvd_long else "CVD미확인",
                ]

        elif bull_4h and long_tf_count == 1 and ema_1h_trend != "below":
            # D급: 4h 단독
            signal = Signal.LONG
            confidence = 0.55
            reason_parts = [
                f"[D급] 강세다이버전스 4h 단독(강도{str_4h})",
                "1h/15m 미확인",
            ]

        elif bull_15m and long_tf_count == 1 and ema_1h_trend != "below":
            # E급: 15m 단독 + CVD 확인 + EMA50 위 필수 (노이즈 방어)
            if cvd_bull_15m and trend_above and str_15m >= 2:
                signal = Signal.LONG
                confidence = 0.60
                reason_parts = [
                    f"[E급] 강세다이버전스 15m(강도{str_15m})",
                    "CVD매수우위",
                    f"EMA{ema_trend_p}위",
                ]

        # ═══════════════ SHORT ═══════════════════════════════════════════
        # ※ signal == NEUTRAL 독립 체크: 약한 LONG 조건 미달 시에도 SHORT 확인
        if signal == Signal.NEUTRAL and short_tf_count >= 2 and ema_1h_trend != "above":
            tf_str = _tf_label(bear_4h, bear_1h, bear_15m)

            if short_tf_count == 3 and cvd_short and not trend_above:
                # A급
                signal = Signal.STRONG_SHORT
                confidence = 0.95
                reason_parts = [
                    f"[A급] 약세다이버전스 {tf_str}(강도{max_short_str})",
                    "CVD매도우위",
                    f"EMA{ema_trend_p}아래",
                ]

            elif bear_4h and bear_1h:
                # B급
                h1_pen = 0.85 if ema_1h_trend == "unknown" else 1.0
                signal = Signal.STRONG_SHORT if cvd_short else Signal.SHORT
                confidence = round(0.82 * h1_pen, 3)
                reason_parts = [
                    f"[B급] 약세다이버전스 {tf_str}(강도{max_short_str})",
                    "CVD매도우위" if cvd_short else "CVD미확인",
                ]

            elif bear_1h and bear_15m:
                # C급
                h1_pen = 0.8 if ema_1h_trend == "above" else 1.0
                signal = Signal.SHORT
                confidence = round(0.72 * h1_pen, 3)
                reason_parts = [
                    f"[C급] 약세다이버전스 {tf_str}(강도{max_short_str})",
                    "CVD매도우위" if cvd_short else "CVD미확인",
                ]

        if signal == Signal.NEUTRAL and bear_4h and short_tf_count == 1 and ema_1h_trend != "above":
            # D급
            signal = Signal.SHORT
            confidence = 0.55
            reason_parts = [
                f"[D급] 약세다이버전스 4h 단독(강도{str_4h})",
                "1h/15m 미확인",
            ]

        if signal == Signal.NEUTRAL and bear_15m and short_tf_count == 1 and ema_1h_trend != "above":
            # E급: 15m 단독 + CVD 확인 + EMA50 아래 필수
            if cvd_bear_15m and not trend_above and str_15m >= 2:
                signal = Signal.SHORT
                confidence = 0.60
                reason_parts = [
                    f"[E급] 약세다이버전스 15m(강도{str_15m})",
                    "CVD매도우위",
                    f"EMA{ema_trend_p}아래",
                ]

        # ── SL / TP ──────────────────────────────────────────────────────
        sl_distance = atr_val * atr_mult

        if signal.value > 0:
            stop_loss   = current_price - sl_distance
            take_profit = current_price + sl_distance * rr_ratio
        elif signal.value < 0:
            stop_loss   = current_price + sl_distance
            take_profit = current_price - sl_distance * rr_ratio
        else:
            stop_loss = take_profit = 0.0

        # 등급 결정
        if signal == Signal.NEUTRAL:
            div_grade = "none"
            active_tf_count = 0
            active_strength = 0
            cvd_hit = False
            logger.info(
                f"🔍 MTF NEUTRAL | "
                f"4h bull={bull_4h}({str_4h}) bear={bear_4h} | "
                f"1h bull={bull_1h}({str_1h}) bear={bear_1h} | "
                f"15m bull={bull_15m}({str_15m}) bear={bear_15m} | "
                f"cvd_long={cvd_long} cvd_short={cvd_short} | "
                f"1h_trend={ema_1h_trend}"
            )
        else:
            is_long = signal.value > 0
            active_tf_count = long_tf_count if is_long else short_tf_count
            active_strength = max_long_str if is_long else max_short_str
            cvd_hit = cvd_long if is_long else cvd_short

            # 등급 텍스트 파싱 (reason_parts 첫 항목에서 추출)
            first = reason_parts[0] if reason_parts else ""
            if "[A급]" in first:
                div_grade = "A"
            elif "[B급]" in first:
                div_grade = "B"
            elif "[C급]" in first:
                div_grade = "C"
            elif "[D급]" in first:
                div_grade = "D"
            elif "[E급]" in first:
                div_grade = "E"
            else:
                div_grade = "none"

            tfs_checked = (
                f"4h[{'✓' if (bull_4h or bear_4h) else '✗'}] "
                f"1h[{'✓' if (bull_1h or bear_1h) else '✗'}] "
                f"15m[{'✓' if (bull_15m or bear_15m) else '✗'}]"
            )
            reason_parts.append(tfs_checked)

        # 분석용 부가 데이터
        extra = {
            "div_grade":    div_grade,
            "div_tf_count": active_tf_count,
            "div_strength": active_strength,
            "cvd_confirmed": cvd_long if signal.value > 0 else (cvd_short if signal.value < 0 else False),
            # TF별 상세
            "bull_4h": bull_4h, "bear_4h": bear_4h, "str_4h": str_4h,
            "bull_1h": bull_1h, "bear_1h": bear_1h, "str_1h": str_1h,
            "bull_15m": bull_15m, "bear_15m": bear_15m, "str_15m": str_15m,
        }

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            reason=" | ".join(reason_parts) if reason_parts else "시그널 없음",
            extra=extra,
        )
