"""시그널 어그리게이터 - 다중 전략의 가중 투표로 최종 시그널 생성"""
import logging
import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from config import Config
from strategy.base import BaseStrategy, StrategyResult, Signal
from strategy.trend_follow import TrendFollowStrategy
from strategy.breakout import BreakoutStrategy
from strategy.scalping import ScalpingStrategy

if TYPE_CHECKING:
    from core.data_feed import DataFeed

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignal:
    """최종 합산 시그널"""
    direction: str  # "long" | "short" | "neutral"
    score: float    # -1.0 ~ 1.0
    confidence: float
    stop_loss: float
    take_profit: float
    reasons: list[str]
    strategy_details: list[dict]
    # ── 3개월 로그 분석용 부가 데이터 ──────────────
    div_grade: str = "none"      # A/B/C/D/E/none
    div_tf_count: int = 0        # 확인된 TF 수 (0~3)
    div_strength: int = 0        # 다이버전스 강도 (0~4)
    cvd_confirmed: bool = False  # CVD 방향 일치 여부
    bb_squeeze: bool = False     # 볼린저 스퀴즈 후 진입 여부
    score_trend: float = 0.0     # trend_follow 개별 기여 점수
    score_breakout: float = 0.0  # breakout 개별 기여 점수


# 전략 이름 → 클래스 매핑
STRATEGY_MAP = {
    "trend_follow": TrendFollowStrategy,
    "breakout": BreakoutStrategy,
    "scalping": ScalpingStrategy,
}


class SignalAggregator:
    """다중 전략 시그널 취합 및 최종 판단"""

    def __init__(self, config: Config):
        self.config = config
        self.strategies: list[tuple[BaseStrategy, float]] = []
        self.signal_threshold = config.strategy.get("signal_threshold", 0.3)
        self._last_log_time: float = 0
        self._log_interval: float = 300  # 5분마다 중립 신호 상태 로깅

        # 전략 인스턴스 생성
        for strat_cfg in config.strategy.get("active", []):
            name = strat_cfg["name"]
            weight = strat_cfg["weight"]
            params = strat_cfg.get("params", {})

            cls = STRATEGY_MAP.get(name)
            if cls:
                self.strategies.append((cls(params), weight))
                logger.info(f"전략 로드: {name} (weight={weight})")
            else:
                logger.warning(f"알 수 없는 전략: {name}")

    def evaluate(self, data_feed: "DataFeed") -> AggregatedSignal:
        """
        모든 전략을 실행하고 가중 투표로 최종 시그널 생성
        """
        primary_tf = self.config.strategy.get("timeframes", {}).get("primary", "15m")
        
        closes = data_feed.get_closes(primary_tf)
        highs = data_feed.get_highs(primary_tf)
        lows = data_feed.get_lows(primary_tf)
        volumes = data_feed.get_volumes(primary_tf)
        current_price = data_feed.get_current_price()

        if len(closes) < 30 or current_price == 0:
            return AggregatedSignal(
                direction="neutral", score=0, confidence=0,
                stop_loss=0, take_profit=0,
                reasons=["데이터 부족"], strategy_details=[],
            )

        rr_ratio = self.config.risk.get("take_profit_rr_ratio", 3.0)

        weighted_score = 0.0
        total_weight = 0.0
        active_weight = 0.0   # 시그널 낸 전략의 weight 합산
        all_sl = []
        all_tp = []
        reasons = []
        details = []
        # 부가 데이터 수집용
        _trend_score = 0.0
        _breakout_score = 0.0
        _trend_extra: dict = {}
        _breakout_extra: dict = {}

        for strategy, weight in self.strategies:
            try:
                # 스캘핑은 오더북 데이터 추가 전달
                extra = {}
                if strategy.name == "scalping":
                    extra["orderbook"] = data_feed.orderbook

                # trend_follow에 멀티 타임프레임 데이터 전달 (15m / 1h / 4h 다이버전스 확인)
                if strategy.name == "trend_follow":
                    from strategy.base import ema as _ema_fn

                    # 1h 데이터
                    closes_1h  = data_feed.get_closes("1h")
                    highs_1h   = data_feed.get_highs("1h")
                    lows_1h    = data_feed.get_lows("1h")
                    volumes_1h = data_feed.get_volumes("1h")

                    # 4h 데이터
                    closes_4h  = data_feed.get_closes("4h")
                    highs_4h   = data_feed.get_highs("4h")
                    lows_4h    = data_feed.get_lows("4h")
                    volumes_4h = data_feed.get_volumes("4h")

                    # 1h EMA 추세 (상위 추세 필터용)
                    ema_1h_trend = "unknown"
                    if len(closes_1h) >= 26:
                        _ef1h = _ema_fn(closes_1h, 9)[-1]
                        _es1h = _ema_fn(closes_1h, 26)[-1]
                        ema_1h_trend = "above" if _ef1h > _es1h else "below"

                    # 거래량 비율 (volumes[-1] 불완전 봉 제외)
                    vol_base = float(np.median(volumes[-21:-1])) if len(volumes) >= 21 else 1.0
                    vol_ratio = float(volumes[-2]) / vol_base if vol_base > 0 else 1.0

                    extra["ema_1h_trend"] = ema_1h_trend
                    extra["vol_ratio"]    = vol_ratio
                    extra["closes_1h"]    = closes_1h
                    extra["highs_1h"]     = highs_1h
                    extra["lows_1h"]      = lows_1h
                    extra["volumes_1h"]   = volumes_1h
                    extra["closes_4h"]    = closes_4h
                    extra["highs_4h"]     = highs_4h
                    extra["lows_4h"]      = lows_4h
                    extra["volumes_4h"]   = volumes_4h

                result: StrategyResult = strategy.analyze(
                    closes, highs, lows, volumes,
                    current_price,
                    rr_ratio=rr_ratio,
                    **extra,
                )

                # 가중 점수 합산
                score = result.signal.value * result.confidence * weight
                weighted_score += score
                total_weight += weight

                # 전략별 개별 점수 추적
                if strategy.name == "trend_follow":
                    _trend_score = score
                    _trend_extra = result.extra
                elif strategy.name == "breakout":
                    _breakout_score = score
                    _breakout_extra = result.extra

                # 시그널을 낸 전략의 weight만 별도 추적 (정규화 희석 방지)
                if result.signal != Signal.NEUTRAL:
                    active_weight += weight
                    reasons.append(f"[{strategy.name}] {result.reason}")
                    if result.stop_loss:
                        all_sl.append(result.stop_loss)
                    if result.take_profit:
                        all_tp.append(result.take_profit)

                details.append({
                    "name": strategy.name,
                    "signal": result.signal.name,
                    "confidence": result.confidence,
                    "weighted_score": score,
                    "reason": result.reason,
                })

            except Exception as e:
                logger.error(f"전략 {strategy.name} 에러: {e}")

        # 정규화: 시그널을 낸 전략의 weight만으로 나눔 (NEUTRAL 전략이 점수 희석하는 버그 수정)
        final_score = weighted_score / active_weight if active_weight > 0 else 0

        # 방향 결정
        if final_score >= self.signal_threshold:
            direction = "long"
        elif final_score <= -self.signal_threshold:
            direction = "short"
        else:
            direction = "neutral"

        # TP/SL: 가장 보수적인 값 선택 (가장 타이트한 손절)
        if direction == "long" and all_sl:
            stop_loss = max(all_sl)   # 롱일 때 가장 높은 SL = 가장 타이트
            take_profit = min(all_tp) if all_tp else 0
        elif direction == "short" and all_sl:
            stop_loss = min(all_sl)   # 숏일 때 가장 낮은 SL = 가장 타이트
            take_profit = max(all_tp) if all_tp else 0
        else:
            stop_loss = take_profit = 0

        confidence = abs(final_score)

        agg = AggregatedSignal(
            direction=direction,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            reasons=reasons,
            strategy_details=details,
            div_grade=_trend_extra.get("div_grade", "none"),
            div_tf_count=_trend_extra.get("div_tf_count", 0),
            div_strength=_trend_extra.get("div_strength", 0),
            cvd_confirmed=bool(_trend_extra.get("cvd_confirmed", False)),
            bb_squeeze=bool(_breakout_extra.get("bb_squeeze", False)),
            score_trend=round(_trend_score, 4),
            score_breakout=round(_breakout_score, 4),
        )

        now = time.time()
        if direction != "neutral":
            logger.info(
                f"🎯 시그널: {direction.upper()} | score={final_score:.3f} | "
                f"SL={stop_loss} TP={take_profit}"
            )
            for r in reasons:
                logger.info(f"   {r}")
            self._last_log_time = now
        elif now - self._last_log_time >= self._log_interval:
            # 5분마다 중립 상태 상세 진단 로그
            self._last_log_time = now
            detail_str = " | ".join(
                f"{d['name']}:{d['signal']}({d['confidence']:.2f})→{d['weighted_score']:+.3f} [{d['reason']}]"
                for d in details
            )
            logger.info(
                f"⏳ 중립 유지 | score={final_score:.3f} (임계값±{self.signal_threshold}) | {detail_str}"
            )

        return agg
