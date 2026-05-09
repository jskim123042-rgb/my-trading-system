"""전략 기본 추상 클래스 및 기술적 지표 유틸리티"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Signal(Enum):
    STRONG_LONG = 1.0
    LONG = 0.6
    NEUTRAL = 0.0
    SHORT = -0.6
    STRONG_SHORT = -1.0


@dataclass
class StrategyResult:
    """전략 시그널 결과"""
    signal: Signal
    confidence: float = 0.0   # 0.0 ~ 1.0
    stop_loss: float = 0.0    # 추천 손절가
    take_profit: float = 0.0  # 추천 익절가
    reason: str = ""
    extra: dict = field(default_factory=dict)  # 전략별 부가 데이터 (로깅/분석용)


class BaseStrategy(ABC):
    """모든 전략의 기본 클래스"""

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def analyze(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        current_price: float,
        **kwargs,
    ) -> StrategyResult:
        """시그널 분석 - 각 전략에서 구현"""
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 기술적 지표 유틸리티 (numpy 기반, 외부 의존 없음)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ema(data: np.ndarray, period: int) -> np.ndarray:
    """지수이동평균"""
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


def sma(data: np.ndarray, period: int) -> np.ndarray:
    """단순이동평균"""
    result = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1 : i + 1])
    return result


def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI (Relative Strength Index)"""
    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros(len(data))
    avg_loss = np.zeros(len(data))

    # 초기 SMA
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # Wilder's smoothing
    for i in range(period + 1, len(data)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    # avg_loss==0이면 RS=inf → RSI=100 (순수 상승장)
    # out=0 패턴은 avg_loss==0 → RS=0 → RSI=0 오류 발생
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi_val = 100 - (100 / (1 + rs))
    rsi_val[:period] = np.nan
    return rsi_val


def macd(
    data: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (line, signal, histogram)"""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    data: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """볼린저 밴드 (upper, middle, lower)"""
    middle = sma(data, period)
    rolling_std = np.full_like(data, np.nan)
    for i in range(period - 1, len(data)):
        rolling_std[i] = np.std(data[i - period + 1 : i + 1])
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return upper, middle, lower


def atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """ATR (Average True Range)"""
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    # Wilder's smoothing
    atr_val = np.zeros(len(closes))
    atr_val[period - 1] = np.mean(tr[:period])
    for i in range(period, len(closes)):
        atr_val[i] = (atr_val[i - 1] * (period - 1) + tr[i]) / period
    atr_val[: period - 1] = np.nan
    return atr_val


def adx(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """ADX (Average Directional Index) — 추세 강도 (0~100)
    < 25: 횡보장 (다이버전스 유리)
    > 25: 추세장
    """
    n = len(closes)
    if n < period * 2:
        return np.full(n, np.nan)

    # True Range
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))

    # +DM / -DM
    pdm = np.zeros(n)
    ndm = np.zeros(n)
    for i in range(1, n):
        up   = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        pdm[i] = up   if (up > down and up > 0)   else 0.0
        ndm[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing
    def _wilder(arr):
        out = np.zeros(n)
        out[period] = arr[1:period + 1].sum()
        for i in range(period + 1, n):
            out[i] = out[i - 1] - out[i - 1] / period + arr[i]
        return out

    atr_w  = _wilder(tr)
    pdm_w  = _wilder(pdm)
    ndm_w  = _wilder(ndm)

    pdi = np.where(atr_w > 0, 100 * pdm_w / atr_w, 0.0)
    ndi = np.where(atr_w > 0, 100 * ndm_w / atr_w, 0.0)

    dx = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)

    # ADX = Wilder smooth of DX
    adx_val = np.zeros(n)
    adx_val[period * 2 - 1] = dx[period:period * 2].mean()
    for i in range(period * 2, n):
        adx_val[i] = (adx_val[i - 1] * (period - 1) + dx[i]) / period
    adx_val[:period * 2 - 1] = np.nan
    return adx_val


def fibonacci_levels(high: float, low: float) -> dict:
    """피보나치 되돌림 레벨"""
    diff = high - low
    return {
        0.0: high,
        0.236: high - diff * 0.236,
        0.382: high - diff * 0.382,
        0.5: high - diff * 0.5,
        0.618: high - diff * 0.618,
        0.786: high - diff * 0.786,
        1.0: low,
    }
