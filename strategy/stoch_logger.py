"""
============================================================
Stoch+RSI+MACD 병행 시그널 로거
============================================================

목적:
  현재 봇(EMA 5/26 + MACD 8/26/7)은 그대로 실매매.
  이 모듈은 Stoch+RSI+MACD 전략의 시그널만 CSV로 기록.
  실제 주문은 하지 않음. 3개월 후 두 전략 성과를 비교하기 위한 용도.

출력 파일:
  data/stoch_signals.csv
"""

import csv
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from strategy.base import rsi, macd, atr

logger = logging.getLogger(__name__)


def stochastic(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    k = np.full(n, np.nan)

    for i in range(k_period - 1, n):
        low_min = np.min(lows[i - k_period + 1: i + 1])
        high_max = np.max(highs[i - k_period + 1: i + 1])
        if high_max - low_min > 0:
            k[i] = (closes[i] - low_min) / (high_max - low_min) * 100
        else:
            k[i] = 50.0

    d = np.full(n, np.nan)
    for i in range(k_period + d_period - 2, n):
        window = k[i - d_period + 1: i + 1]
        if not np.any(np.isnan(window)):
            d[i] = np.mean(window)

    return k, d


def evaluate_stoch_signal(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    current_index: int,
) -> dict:
    i = current_index
    if i < 30 or i >= len(closes):
        return _neutral_result(0, 0, 0, 0, 0, closes[i] if i < len(closes) else 0)

    k, d = stochastic(highs, lows, closes, 14, 3)
    rsi_v = rsi(closes, 14)
    _, _, hist = macd(closes, 12, 26, 9)
    atr_v = atr(highs, lows, closes, 14)

    ck, cd, cr, ch, ca, cc = k[i], d[i], rsi_v[i], hist[i], atr_v[i], closes[i]

    if np.isnan(ck) or np.isnan(cd) or np.isnan(cr) or np.isnan(ca):
        return _neutral_result(ck, cd, cr, ch, ca, cc)

    stoch_cross_up = k[i] > d[i] and k[i - 1] <= d[i - 1]
    stoch_cross_dn = k[i] < d[i] and k[i - 1] >= d[i - 1]
    oversold_exit  = k[i] > 20 and k[i - 1] <= 20
    overbought_exit = k[i] < 80 and k[i - 1] >= 80

    macd_bull = hist[i] > 0 and hist[i - 1] <= 0
    macd_bear = hist[i] < 0 and hist[i - 1] >= 0
    macd_pos  = hist[i] > 0
    macd_neg  = hist[i] < 0

    sl_dist  = ca * 1.5
    sl_long  = cc - sl_dist;  tp_long  = cc + sl_dist * 3
    sl_short = cc + sl_dist;  tp_short = cc - sl_dist * 3

    signal, signal_type, confidence, reason = "neutral", "NEUTRAL", 0.0, ""

    if stoch_cross_up and 50 < cr < 75 and macd_bull:
        signal, signal_type, confidence = "long", "STRONG_LONG", 0.95
        reason = f"Stoch GC({ck:.0f}/{cd:.0f}) + RSI={cr:.0f} + MACD 양전환"
    elif stoch_cross_up and 50 < cr < 75 and macd_pos:
        signal, signal_type, confidence = "long", "LONG", 0.80
        reason = f"Stoch GC({ck:.0f}/{cd:.0f}) + RSI={cr:.0f} + MACD 양수"
    elif oversold_exit and cr > 45 and macd_pos:
        signal, signal_type, confidence = "long", "LONG", 0.75
        reason = f"Stoch 과매도탈출({ck:.0f}) + RSI={cr:.0f} + MACD 양수"
    elif stoch_cross_dn and 25 < cr < 50 and macd_bear:
        signal, signal_type, confidence = "short", "STRONG_SHORT", 0.95
        reason = f"Stoch DC({ck:.0f}/{cd:.0f}) + RSI={cr:.0f} + MACD 음전환"
    elif stoch_cross_dn and 25 < cr < 50 and macd_neg:
        signal, signal_type, confidence = "short", "SHORT", 0.80
        reason = f"Stoch DC({ck:.0f}/{cd:.0f}) + RSI={cr:.0f} + MACD 음수"
    elif overbought_exit and cr < 55 and macd_neg:
        signal, signal_type, confidence = "short", "SHORT", 0.75
        reason = f"Stoch 과매수이탈({ck:.0f}) + RSI={cr:.0f} + MACD 음수"

    return {
        "signal": signal,
        "signal_type": signal_type,
        "confidence": confidence,
        "stoch_k": round(ck, 1),
        "stoch_d": round(cd, 1),
        "rsi": round(cr, 1),
        "macd_hist": round(ch, 4),
        "atr": round(ca, 2),
        "reason": reason or "조건 미충족",
        "entry_price": round(cc, 2),
        "sl": round(sl_long  if signal == "long"  else sl_short if signal == "short" else 0, 2),
        "tp": round(tp_long  if signal == "long"  else tp_short if signal == "short" else 0, 2),
    }


def _neutral_result(ck, cd, cr, ch, ca, cc):
    def _s(v): return round(v, 1) if not (isinstance(v, float) and np.isnan(v)) else 0
    return {
        "signal": "neutral", "signal_type": "NEUTRAL", "confidence": 0.0,
        "stoch_k": _s(ck), "stoch_d": _s(cd), "rsi": _s(cr),
        "macd_hist": round(ch, 4) if not np.isnan(ch) else 0,
        "atr": round(ca, 2) if not np.isnan(ca) else 0,
        "reason": "데이터 부족 또는 조건 미충족",
        "entry_price": round(cc, 2), "sl": 0, "tp": 0,
    }


CSV_HEADERS = [
    "timestamp", "price", "signal", "signal_type", "confidence",
    "stoch_k", "stoch_d", "rsi", "macd_hist", "atr",
    "sl", "tp", "reason",
    "current_ema5", "current_ema26", "current_signal",
]


class StochSignalLogger:
    """
    Stoch+RSI+MACD 시그널을 CSV로 기록.
    실매매 없음. 3개월 후 현재 EMA 전략과 비교 분석용.
    """

    def __init__(self, output_path: str = "data/stoch_signals.csv"):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_ts: str = ""   # 중복 기록 방지

        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADERS)

        logger.info(f"📝 StochSignalLogger 초기화 | {self.path}")

    def log(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        current_ema5: float = 0.0,
        current_ema26: float = 0.0,
        current_signal: str = "",
    ):
        """매 15분봉 마감 시 호출."""
        if len(closes) < 30:
            return

        # 같은 캔들 중복 기록 방지 (5초 이내 재호출)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        if ts == self._last_ts:
            return
        self._last_ts = ts

        i = len(closes) - 1
        result = evaluate_stoch_signal(closes, highs, lows, volumes, i)

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                result["entry_price"],
                result["signal"],
                result["signal_type"],
                result["confidence"],
                result["stoch_k"],
                result["stoch_d"],
                result["rsi"],
                result["macd_hist"],
                result["atr"],
                result["sl"],
                result["tp"],
                result["reason"],
                round(current_ema5, 2),
                round(current_ema26, 2),
                current_signal,
            ])

        if result["signal"] != "neutral":
            logger.info(
                f"🔮 [Stoch] {result['signal_type']} | "
                f"%K={result['stoch_k']} %D={result['stoch_d']} | "
                f"RSI={result['rsi']} | {result['reason']}"
            )
