"""
Bitget 선물 자동매매 에이전트 - 메인 루프
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

실행: python main.py [--config config/settings.yaml]

흐름:
1. 데이터 수집 (WebSocket + REST)
2. 전략 엔진 → 시그널 생성
3. 리스크 관리 → 진입 가능 여부 + 포지션 사이징
4. 주문 실행 → TP/SL 동시 설정
5. 포지션 모니터링 → 트레일링 스탑 전환
"""
import argparse
import asyncio
import logging
import logging.handlers
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import load_config, Config
from core.client import BitgetClient
from core.data_feed import DataFeed
from core.order_manager import OrderManager, Side
from strategy.aggregator import SignalAggregator
from strategy.base import atr as calc_atr
from risk.risk_manager import PositionSizer, DrawdownGuard, LeverageController
from monitor.telegram_bot import TelegramNotifier
from monitor.trade_logger import TradeLogger
from strategy.stoch_logger import StochSignalLogger


def setup_logging(config: Config):
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    level = getattr(logging, config.logging.get("level", "INFO"))
    fmt = "%(asctime)s | %(levelname)-5s | %(name)-20s | %(message)s"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            stream_handler,
            logging.handlers.RotatingFileHandler(
                config.logging.get("file", "logs/trading.log"),
                maxBytes=config.logging.get("max_size_mb", 50) * 1024 * 1024,
                backupCount=config.logging.get("backup_count", 10),
                encoding="utf-8",
            ),
        ],
    )


logger = logging.getLogger("agent")


class TradingAgent:
    """메인 트레이딩 에이전트"""

    def __init__(self, config: Config):
        self.config = config

        # 모듈 초기화
        self.client = BitgetClient(config)
        self.data_feed = DataFeed(config, self.client)
        self.order_mgr = OrderManager(config, self.client)
        self.aggregator = SignalAggregator(config)
        self.position_sizer = PositionSizer(config)
        self.drawdown_guard = DrawdownGuard(config)
        self.leverage_ctrl = LeverageController(config)
        self.notifier = TelegramNotifier(config)
        self.trade_logger = TradeLogger()
        self.stoch_logger = StochSignalLogger("data/stoch_signals.csv")

        # 상태
        self._running = False
        self._last_eval_time = 0
        self._eval_interval = 5  # 시그널 평가 간격 (초)
        self._last_daily_reset = ""
        self._last_status_log_time = 0
        self._status_log_interval = 300  # 5분마다 루프 상태 로그
        self._last_indicator_log_time = 0
        self._last_signal = None  # 마지막 신호 평가 결과 (지표 알림용)
        self._indicator_log_interval = 900  # 15분마다 지표 현황 로그
        self._last_market_fetch_time = 0
        self._market_fetch_interval = 60   # 60초마다 OI/롱숏 REST 조회
        self._last_vol_profile_time = 0
        self._vol_profile_interval = 300   # 5분마다 거래량 프로파일 스냅샷
        # Shadow trade 추적 (포지션 중 무시된 반대 시그널 가상 추적)
        self._shadow_trades: list[dict] = []  # {id, direction, entry, sl, tp, open_time}

    async def start(self):
        """에이전트 시작"""
        logger.info("=" * 60)
        logger.info("🤖 Bitget 선물 자동매매 에이전트 시작")
        logger.info("=" * 60)

        # 잔고 확인
        balance = self.client.get_balance()
        logger.info(f"💰 계좌 잔고: {balance['total']:.2f} USDT (가용: {balance['free']:.2f})")

        # 드로다운 가드 초기화
        self.drawdown_guard.reset_daily(balance["total"])

        # 재시작 시 누적 거래 히스토리 DB에서 복원 (승률/켈리 연속성 유지)
        history = self.trade_logger.load_trade_history()
        self.order_mgr.trade_history.extend(history)

        # 재시작 시 기존 포지션 복구
        restored = self.order_mgr.sync_position()
        if restored:
            trade = self.order_mgr.current_trade
            await self.notifier.send(
                f"♻️ <b>포지션 복구</b>\n"
                f"방향: {'LONG' if trade.side == 'buy' else 'SHORT'}\n"
                f"수량: {trade.amount} BTC\n"
                f"진입가: ${trade.entry_price:,.2f}\n"
                f"손절: ${trade.stop_loss:,.2f}\n"
                f"익절: ${trade.take_profit:,.2f}"
            )

        # 텔레그램 알림
        await self.notifier.send(
            f"🤖 <b>에이전트 시작</b>\n잔고: ${balance['total']:,.2f} USDT"
        )

        self._running = True

        # 데이터 수집 + 메인 루프 동시 실행
        await asyncio.gather(
            self.data_feed.start(),
            self._main_loop(),
        )

    async def _main_loop(self):
        """메인 트레이딩 루프"""
        # 초기 데이터 로드 대기
        await asyncio.sleep(3)

        while self._running:
            try:
                now = time.time()

                # 일간 리셋 체크
                self._check_daily_reset()

                # 60초마다 OI / 롱숏 REST 조회
                if now - self._last_market_fetch_time >= self._market_fetch_interval:
                    self._last_market_fetch_time = now
                    self.data_feed.fetch_market_context()

                # 15분마다 지표 현황 로그 + 텔레그램
                if now - self._last_indicator_log_time >= self._indicator_log_interval:
                    self._last_indicator_log_time = now
                    current_price_for_log = self.data_feed.get_current_price()
                    if current_price_for_log > 0:
                        await self._log_indicators(current_price_for_log)

                # 5분마다 거래량 프로파일 스냅샷
                if now - self._last_vol_profile_time >= self._vol_profile_interval:
                    self._last_vol_profile_time = now
                    _snap_price = self.data_feed.get_current_price()
                    if _snap_price > 0:
                        primary_tf = self.config.strategy.get("timeframes", {}).get("primary", "15m")
                        _vs = self.data_feed.get_volume_stats(primary_tf)
                        _vd = self.data_feed.get_vol_delta(primary_tf)
                        if _vs:
                            self.trade_logger.log_volume_profile(
                                price=_snap_price,
                                vol_stats=_vs,
                                vol_delta=_vd,
                                oi=self.data_feed.oi,
                                oi_change_pct=self.data_feed.get_oi_change_pct(),
                                long_short_ratio=self.data_feed.long_short_ratio,
                            )

                # 평가 간격 체크
                if now - self._last_eval_time < self._eval_interval:
                    await asyncio.sleep(0.5)
                    continue

                self._last_eval_time = now

                # 현재 가격
                current_price = self.data_feed.get_current_price()
                if current_price == 0:
                    logger.warning("⚠️ 현재가 0 — 데이터 수신 대기 중")
                    await asyncio.sleep(1)
                    continue

                # 전략 파라미터 (청산 시 지표 추출에도 공용)
                primary_tf = self.config.strategy.get("timeframes", {}).get("primary", "15m")
                _strat_p = next(
                    (s.get("params", {}) for s in self.config.strategy.get("active", [])
                     if s["name"] == "trend_follow"), {}
                )

                # 포지션 보유 중이면 모니터링만
                if self.order_mgr.has_position:
                    # MFE/MAE DB 업데이트 (30초마다)
                    t = self.order_mgr.current_trade
                    if t and t.id and hasattr(self, '_last_mfe_update'):
                        if now - self._last_mfe_update >= 30:
                            self._last_mfe_update = now
                            self.trade_logger.update_excursion(t.id, t.mfe_pct, t.mae_pct, t.mfe_r)
                    elif not hasattr(self, '_last_mfe_update'):
                        self._last_mfe_update = now

                    # ── Shadow trade SL/TP 가상 체결 체크 ──
                    still_open = []
                    for st in self._shadow_trades:
                        if st["direction"] == "long":
                            hit_sl = current_price <= st["sl"]
                            hit_tp = current_price >= st["tp"]
                        else:
                            hit_sl = current_price >= st["sl"]
                            hit_tp = current_price <= st["tp"]
                        if hit_sl or hit_tp:
                            outcome = "tp" if hit_tp else "sl"
                            ep = st["tp"] if hit_tp else st["sl"]
                            if st["direction"] == "long":
                                vpnl = (ep - st["entry"]) / st["entry"]
                            else:
                                vpnl = (st["entry"] - ep) / st["entry"]
                            st_leverage = st.get("leverage", self.config.leverage["default"])
                            dur = (time.time() - st["open_time"]) / 60
                            self.trade_logger.update_shadow_trade(
                                row_id=st["id"],
                                outcome=outcome,
                                exit_price=ep,
                                virtual_pnl_pct=vpnl * st_leverage,
                                duration_min=dur,
                            )
                            logger.info(
                                f"👻 Shadow trade 종료 | {st['direction'].upper()} "
                                f"{'✅TP' if outcome=='tp' else '❌SL'} | "
                                f"가상손익={vpnl * st_leverage:+.2%}"
                            )
                        else:
                            still_open.append(st)
                    self._shadow_trades = still_open

                    # ── 반대 시그널 발생 시 shadow trade 기록 ──
                    _shadow_signal = self.aggregator.evaluate(self.data_feed)
                    t = self.order_mgr.current_trade
                    if t and _shadow_signal.direction != "neutral":
                        is_opposite = (
                            (t.side == "buy" and _shadow_signal.direction == "short") or
                            (t.side == "sell" and _shadow_signal.direction == "long")
                        )
                        if is_opposite and _shadow_signal.stop_loss > 0:
                            # 현재 포지션 미실현 손익
                            if t.side == "buy":
                                _cur_pnl = (current_price - t.entry_price) / t.entry_price
                            else:
                                _cur_pnl = (t.entry_price - current_price) / t.entry_price
                            _cur_pnl_lev = _cur_pnl * t.leverage
                            row_id = self.trade_logger.log_shadow_trade(
                                direction=_shadow_signal.direction,
                                score=_shadow_signal.score,
                                signal_price=current_price,
                                stop_loss=_shadow_signal.stop_loss,
                                take_profit=_shadow_signal.take_profit,
                                reasons=_shadow_signal.reasons,
                                active_side=t.side,
                                active_entry=t.entry_price,
                                active_pnl_pct=_cur_pnl_lev,
                            )
                            if row_id > 0:
                                self._shadow_trades.append({
                                    "id": row_id,
                                    "direction": _shadow_signal.direction,
                                    "entry": current_price,
                                    "sl": _shadow_signal.stop_loss,
                                    "tp": _shadow_signal.take_profit,
                                    "open_time": time.time(),
                                    "leverage": self.leverage_ctrl.get_leverage(),
                                })
                                logger.info(
                                    f"👻 Shadow trade 기록 | {_shadow_signal.direction.upper()} "
                                    f"score={_shadow_signal.score:.3f} | "
                                    f"SL={_shadow_signal.stop_loss} TP={_shadow_signal.take_profit}"
                                )

                    closed = self.order_mgr.check_position(current_price)
                    if closed:
                        duration_min = (time.time() - closed.entry_time) / 60
                        # ── 청산 시점 지표 추출 ──
                        _exit_ema_trend = "unknown"
                        _exit_macd_hist = _exit_rsi = _exit_atr = _exit_vol_ratio = 0.0
                        try:
                            _ec = self.data_feed.get_closes(primary_tf)
                            _eh = self.data_feed.get_highs(primary_tf)
                            _el = self.data_feed.get_lows(primary_tf)
                            _ev = self.data_feed.get_volumes(primary_tf)
                            if len(_ec) >= 30:
                                from strategy.base import ema as _ema, rsi as _rsi, macd as _macd
                                _ef_x = _ema(_ec, _strat_p.get("ema_fast", 5))[-1]
                                _es_x = _ema(_ec, _strat_p.get("ema_slow", 26))[-1]
                                _exit_ema_trend = "above" if _ef_x > _es_x else "below"
                                _, _, _mh = _macd(_ec, _strat_p.get("macd_fast", 8),
                                                   _strat_p.get("macd_slow", 26),
                                                   _strat_p.get("macd_signal", 7))
                                _exit_macd_hist = float(_mh[-1])
                                _exit_rsi = float(_rsi(_ec, 14)[-1])
                                _exit_atr = float(calc_atr(_eh, _el, _ec, 14)[-1])
                                if len(_ev) >= 20 and _ev[-20:].mean() > 0:
                                    _exit_vol_ratio = float(_ev[-1]) / float(_ev[-20:].mean())
                        except Exception:
                            pass
                        # ── 청산 DB 기록 ──
                        self.trade_logger.log_trade_close(
                            trade_id=closed.id,
                            exit_price=closed.exit_price,
                            pnl_usdt=closed.pnl,
                            pnl_pct=closed.pnl_pct,
                            exit_reason=closed.exit_reason,
                            duration_min=duration_min,
                            exit_ema_trend=_exit_ema_trend,
                            exit_macd_hist=_exit_macd_hist,
                            exit_rsi=_exit_rsi,
                            exit_atr=_exit_atr,
                            exit_vol_ratio=_exit_vol_ratio,
                        )
                        # ── 누적 통계 (DB 기반, 재시작해도 유지) ──
                        cumulative = self.trade_logger.get_cumulative_stats()
                        _bal_after = self.client.get_balance()
                        await self.notifier.notify_exit(
                            side=closed.side,
                            entry_price=closed.entry_price,
                            exit_price=closed.exit_price,
                            pnl_pct=closed.pnl_pct,
                            pnl_usdt=closed.pnl,
                            reason=closed.exit_reason,
                            duration_min=duration_min,
                            cumulative_stats=cumulative,
                            balance=_bal_after["total"],
                            leverage=closed.leverage,
                            mfe_r=closed.mfe_r,
                        )
                    await asyncio.sleep(1)
                    continue

                # 트레이딩 가능 여부 확인
                balance = self.client.get_balance()
                can_trade, reason = self.drawdown_guard.can_trade(balance["total"])
                if not can_trade:
                    logger.info(f"⏸️ 트레이딩 일시중지: {reason}")
                    await asyncio.sleep(30)
                    continue

                # ── 시그널 평가 ──
                signal = self.aggregator.evaluate(self.data_feed)
                self._last_signal = signal  # 지표 알림에서 다이버전스 상태 표시용

                # ── 지표 추출 (신호 로깅용) ──
                highs = self.data_feed.get_highs(primary_tf)
                lows = self.data_feed.get_lows(primary_tf)
                closes = self.data_feed.get_closes(primary_tf)
                volumes = self.data_feed.get_volumes(primary_tf)

                _ema_f = _ema_s = _macd_h = _macd_p = _rsi_v = _atr_v = 0.0
                _ema_1h_trend = _ema_4h_trend = "unknown"
                _rsi_1h = _vol_ratio = 0.0
                _vol_stats: dict = {}
                _vol_delta: dict = {}
                if len(closes) >= 30:
                    from strategy.base import ema as _ema, rsi as _rsi, macd as _macd
                    _ema_f = _ema(closes, _strat_p.get("ema_fast", 5))[-1]
                    _ema_s = _ema(closes, _strat_p.get("ema_slow", 26))[-1]
                    _, _, _hist = _macd(closes, _strat_p.get("macd_fast", 8),
                                        _strat_p.get("macd_slow", 26),
                                        _strat_p.get("macd_signal", 7))
                    _macd_h, _macd_p = _hist[-1], _hist[-2]
                    _rsi_v = _rsi(closes, _strat_p.get("rsi_period", 14))[-1]
                if len(closes) >= 14:
                    _atr_v = calc_atr(highs, lows, closes, 14)[-1]
                # 거래량 통계
                if len(volumes) >= 20:
                    _vol_ratio = float(volumes[-1]) / (sum(volumes[-20:]) / 20) if volumes[-20:].mean() > 0 else 0
                    _vol_stats = self.data_feed.get_volume_stats(primary_tf)
                    _vol_delta = self.data_feed.get_vol_delta(primary_tf)
                # 1h 상위 추세
                closes_1h = self.data_feed.get_closes("1h")
                if len(closes_1h) >= 26:
                    _ef1h = _ema(closes_1h, 9)[-1]
                    _es1h = _ema(closes_1h, 26)[-1]
                    _ema_1h_trend = "above" if _ef1h > _es1h else "below"
                    _rsi_1h = _rsi(closes_1h, 14)[-1]
                # 4h 상위 추세 + RSI
                _rsi_4h = 0.0
                closes_4h = self.data_feed.get_closes("4h")
                if len(closes_4h) >= 26:
                    _ef4h = _ema(closes_4h, 9)[-1]
                    _es4h = _ema(closes_4h, 26)[-1]
                    _ema_4h_trend = "above" if _ef4h > _es4h else "below"
                    _rsi_4h = float(_rsi(closes_4h, 14)[-1])

                # 진입 캔들 강도 (몸통 / 전체 범위 비율)
                _candle_body_pct = 0.0
                if len(closes) >= 2 and len(highs) >= 2 and len(lows) >= 2:
                    _body = abs(closes[-2] - (closes[-3] if len(closes) >= 3 else closes[-2]))
                    _rng = highs[-2] - lows[-2]
                    _candle_body_pct = round(_body / _rng, 3) if _rng > 0 else 0.0

                if signal.direction == "neutral":
                    # 5분마다 루프 정상 동작 확인 로그 + NEUTRAL 신호 DB 기록
                    if now - self._last_status_log_time >= self._status_log_interval:
                        self._last_status_log_time = now
                        balance = self.client.get_balance()
                        can, reason = self.drawdown_guard.can_trade(balance["total"])
                        logger.info(
                            f"💡 루프 정상 동작 | 가격=${current_price:,.0f} | "
                            f"잔고={balance['total']:.2f} USDT | 거래가능={can} ({reason})"
                        )
                        # 5분마다 NEUTRAL 상태도 기록 (나중에 "왜 못 들어갔나" 분석용)
                        if _atr_v > 0:
                            self.trade_logger.log_signal(
                                price=current_price,
                                ema_fast=_ema_f, ema_slow=_ema_s,
                                macd_hist=_macd_h, macd_prev=_macd_p,
                                rsi=_rsi_v, atr=_atr_v,
                                score=signal.score,
                                direction="neutral",
                                fired=False,
                                reasons=[d["reason"] for d in signal.strategy_details],
                                ema_1h_trend=_ema_1h_trend,
                                rsi_1h=_rsi_1h,
                                volume_ratio=_vol_ratio,
                                funding_rate=self.data_feed.funding_rate,
                                vol_current=_vol_stats.get("vol_current", 0),
                                vol_sma20=_vol_stats.get("vol_sma20", 0),
                                vol_spike=_vol_stats.get("vol_spike", False),
                                vol_dry=_vol_stats.get("vol_dry", False),
                                oi_current=self.data_feed.oi,
                                oi_change_1h_pct=self.data_feed.get_oi_change_pct(),
                                long_short_ratio=self.data_feed.long_short_ratio,
                                buy_vol_pct=_vol_stats.get("buy_vol_pct", 0),
                                vol_delta=_vol_delta.get("vol_delta", 0),
                                cvd_slope_5=_vol_delta.get("cvd_slope_5", 0),
                                div_grade=signal.div_grade,
                                div_tf_count=signal.div_tf_count,
                                div_strength=signal.div_strength,
                                cvd_confirmed=signal.cvd_confirmed,
                                bb_squeeze=signal.bb_squeeze,
                                rsi_4h=_rsi_4h,
                                score_trend=signal.score_trend,
                                score_breakout=signal.score_breakout,
                            )
                    await asyncio.sleep(1)
                    continue

                # ── 신호 발생 DB 기록 (진입 전) ──
                self.trade_logger.log_signal(
                    price=current_price,
                    ema_fast=_ema_f, ema_slow=_ema_s,
                    macd_hist=_macd_h, macd_prev=_macd_p,
                    rsi=_rsi_v, atr=_atr_v,
                    score=signal.score,
                    direction=signal.direction,
                    fired=False,
                    reasons=signal.reasons,
                    ema_1h_trend=_ema_1h_trend,
                    rsi_1h=_rsi_1h,
                    volume_ratio=_vol_ratio,
                    funding_rate=self.data_feed.funding_rate,
                    vol_current=_vol_stats.get("vol_current", 0),
                    vol_sma20=_vol_stats.get("vol_sma20", 0),
                    vol_spike=_vol_stats.get("vol_spike", False),
                    vol_dry=_vol_stats.get("vol_dry", False),
                    oi_current=self.data_feed.oi,
                    oi_change_1h_pct=self.data_feed.get_oi_change_pct(),
                    long_short_ratio=self.data_feed.long_short_ratio,
                    buy_vol_pct=_vol_stats.get("buy_vol_pct", 0),
                    vol_delta=_vol_delta.get("vol_delta", 0),
                    cvd_slope_5=_vol_delta.get("cvd_slope_5", 0),
                    div_grade=signal.div_grade,
                    div_tf_count=signal.div_tf_count,
                    div_strength=signal.div_strength,
                    cvd_confirmed=signal.cvd_confirmed,
                    bb_squeeze=signal.bb_squeeze,
                    rsi_4h=_rsi_4h,
                    score_trend=signal.score_trend,
                    score_breakout=signal.score_breakout,
                )

                # ── 변동성 기반 레버리지 조절 ──
                if len(closes) >= 14:
                    leverage = self.leverage_ctrl.adjust_for_volatility(_atr_v, current_price)
                else:
                    leverage = self.config.leverage["default"]

                # ── 등급 기반 레버리지 추가 조절 (ATR 조절 위에 叠加) ──
                leverage = self.leverage_ctrl.adjust_for_signal_grade(signal.div_grade)
                self.client.set_leverage(leverage)

                # ── 등급 기반 R:R → TP 재계산 ──
                _grade_rr_map: dict = self.config.risk.get("grade_rr_ratios", {})
                _base_rr: float = self.config.risk.get("take_profit_rr_ratio", 3.0)
                _rr_ratio: float = _grade_rr_map.get(signal.div_grade, _base_rr)
                _sl_dist = abs(current_price - signal.stop_loss)
                if _sl_dist > 0:
                    if signal.direction == "long":
                        signal.take_profit = round(current_price + _sl_dist * _rr_ratio, 2)
                    else:
                        signal.take_profit = round(current_price - _sl_dist * _rr_ratio, 2)
                logger.info(
                    f"📐 등급 기반 조정 | {signal.div_grade}급 | "
                    f"레버리지={leverage}x | R:R=1:{_rr_ratio} | "
                    f"SL={signal.stop_loss} TP={signal.take_profit}"
                )

                # ── 포지션 사이징 ──
                stats = self.order_mgr.get_stats()
                amount = self.position_sizer.calculate_size(
                    balance=balance["free"],
                    entry_price=current_price,
                    stop_loss=signal.stop_loss,
                    leverage=leverage,
                    win_rate=stats.get("win_rate", 0),
                    avg_rr=stats.get("profit_factor", 0),
                    total_trades=stats.get("total", 0),
                )

                if amount <= 0:
                    continue

                # ── 주문 실행 ──
                side = Side.LONG if signal.direction == "long" else Side.SHORT

                trade = self.order_mgr.open_position(
                    side=side,
                    amount=amount,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    current_price=current_price,
                    entry_reasons=signal.reasons,
                    leverage=leverage,
                )

                if not trade:
                    # 주문 실패 시 30초 쿨다운 (재시도 스팸 방지)
                    self._last_eval_time = now + 30 - self._eval_interval
                    await asyncio.sleep(5)
                    continue

                if trade:
                    # ── 진입 성공 DB 기록 ──
                    self.trade_logger.log_trade_open(
                        trade_id=trade.id,
                        side=trade.side,
                        entry_price=trade.entry_price,
                        amount=trade.amount,
                        stop_loss=trade.stop_loss,
                        take_profit=trade.take_profit,
                        leverage=leverage,
                        ema_fast=_ema_f, ema_slow=_ema_s,
                        macd_hist=_macd_h, macd_prev=_macd_p,
                        rsi=_rsi_v, atr=_atr_v,
                        score=signal.score,
                        reasons=signal.reasons,
                        ema_1h_trend=_ema_1h_trend,
                        ema_4h_trend=_ema_4h_trend,
                        rsi_1h=_rsi_1h,
                        volume_ratio=_vol_ratio,
                        funding_rate=self.data_feed.funding_rate,
                        vol_current=_vol_stats.get("vol_current", 0),
                        vol_sma20=_vol_stats.get("vol_sma20", 0),
                        vol_trend_5=_vol_stats.get("vol_trend_5", 0),
                        buy_vol_pct=_vol_stats.get("buy_vol_pct", 0),
                        oi_current=self.data_feed.oi,
                        oi_change_pct=self.data_feed.get_oi_change_pct(),
                        long_short_ratio=self.data_feed.long_short_ratio,
                        vol_delta=_vol_delta.get("vol_delta", 0),
                        cvd_slope_5=_vol_delta.get("cvd_slope_5", 0),
                        signal_price=current_price,
                        div_grade=signal.div_grade,
                        div_tf_count=signal.div_tf_count,
                        div_strength=signal.div_strength,
                        cvd_confirmed=signal.cvd_confirmed,
                        entry_bb_squeeze=signal.bb_squeeze,
                        entry_rsi_4h=_rsi_4h,
                        score_trend=signal.score_trend,
                        score_breakout=signal.score_breakout,
                        candle_body_pct=_candle_body_pct,
                    )
                    await self.notifier.notify_entry(
                        side=trade.side,
                        amount=trade.amount,
                        price=trade.entry_price,
                        sl=trade.stop_loss,
                        tp=trade.take_profit,
                        reasons=signal.reasons,
                        score=signal.score,
                        leverage=trade.leverage,
                        div_grade=signal.div_grade,
                        div_tf_count=signal.div_tf_count,
                        rr_ratio=_rr_ratio,
                    )

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"메인 루프 에러: {e}", exc_info=True)
                await self.notifier.notify_error(str(e))
                await asyncio.sleep(5)

    async def _log_indicators(self, current_price: float):
        """15분마다 주요 지표 현황 출력 + 텔레그램 발송"""
        from strategy.base import ema, rsi, macd
        primary_tf = self.config.strategy.get("timeframes", {}).get("primary", "15m")
        closes = self.data_feed.get_closes(primary_tf)
        if len(closes) < 30:
            return

        strat_params = {}
        for s in self.config.strategy.get("active", []):
            if s["name"] == "trend_follow":
                strat_params = s.get("params", {})
                break

        ema_fast_arr = ema(closes, strat_params.get("ema_fast", 5))
        ema_slow_arr = ema(closes, strat_params.get("ema_slow", 26))
        rsi_val = rsi(closes, strat_params.get("rsi_period", 14))[-1]
        _, _, histogram = macd(
            closes,
            strat_params.get("macd_fast", 8),
            strat_params.get("macd_slow", 26),
            strat_params.get("macd_signal", 7),
        )

        ema_f = ema_fast_arr[-1]
        ema_s = ema_slow_arr[-1]
        hist_curr = histogram[-1]
        hist_prev = histogram[-2]
        trend = "📈 정배열" if ema_f > ema_s else "📉 역배열"
        macd_state = (
            "🟢 양전환" if hist_prev < 0 and hist_curr > 0 else
            "🔴 음전환" if hist_prev > 0 and hist_curr < 0 else
            "▲ 양수" if hist_curr > 0 else
            "▼ 음수"
        )
        improving = "↑개선" if hist_curr > hist_prev else "↓악화"

        # 현재 포지션 상태
        if self.order_mgr.has_position:
            t = self.order_mgr.current_trade
            direction = "LONG 📈" if t.side == "buy" else "SHORT 📉"
            unrealized = (
                (current_price - t.entry_price) / t.entry_price
                if t.side == "buy"
                else (t.entry_price - current_price) / t.entry_price
            ) * t.leverage
            position_line = f"  포지션: {direction} @ ${t.entry_price:,.1f} | 평가손익: {unrealized:+.2%}\n"
        else:
            position_line = "  포지션: 없음\n"

        logger.info(
            f"━━ 15분 지표 현황 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  BTC:  ${current_price:,.1f}\n"
            f"{position_line}"
            f"  EMA:  fast={ema_f:,.1f}  slow={ema_s:,.1f}  → {trend}\n"
            f"  MACD: hist={hist_curr:+.2f} (전봉={hist_prev:+.2f})  {macd_state} {improving}\n"
            f"  RSI:  {rsi_val:.1f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        # 다이버전스 상태 (실제 트리거 반영)
        sig = self._last_signal
        await self.notifier.notify_indicators(
            price=current_price,
            ema_fast=ema_f,
            ema_slow=ema_s,
            hist_curr=hist_curr,
            hist_prev=hist_prev,
            rsi_val=rsi_val,
            last_signal=sig,
        )

    def _check_daily_reset(self):
        """일간 상태 리셋 (UTC 00:00)"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._last_daily_reset:
            # 전일 스냅샷 저장
            if self._last_daily_reset:
                try:
                    balance = self.client.get_balance()
                    stats = self.order_mgr.get_stats()
                    self.trade_logger.log_daily(
                        date=self._last_daily_reset,
                        start_balance=self.drawdown_guard.state.day_start_balance,
                        end_balance=balance["total"],
                        stats=stats,
                    )
                except Exception:
                    pass

            self._last_daily_reset = today
            balance = self.client.get_balance()
            self.drawdown_guard.reset_daily(balance["total"])

            # 일간 요약 발송
            stats = self.order_mgr.get_stats()
            try:
                _daily_bal = self.client.get_balance()["total"]
            except Exception:
                _daily_bal = 0.0
            asyncio.ensure_future(self.notifier.notify_daily_summary(stats, balance=_daily_bal))

            # R:R 최적화 리포트 (누적 20건 이상일 때만)
            try:
                cum = self.trade_logger.get_cumulative_stats()
                if cum.get("total", 0) >= 20:
                    rr_log = self.trade_logger.get_rr_summary_log()
                    logger.info(rr_log)
            except Exception:
                pass

    async def stop(self):
        """에이전트 종료"""
        logger.info("🛑 에이전트 종료 중...")
        self._running = False
        await self.data_feed.stop()

        # 열린 포지션 경고
        if self.order_mgr.has_position:
            logger.warning("⚠️ 열린 포지션이 있습니다! 수동 확인 필요")
            await self.notifier.send("⚠️ 에이전트 종료 - 열린 포지션 수동 확인 필요")

        stats = self.order_mgr.get_stats()
        logger.info(f"📊 최종 통계: {stats}")
        await self.notifier.send(f"🛑 에이전트 종료\n{stats}")


async def main():
    parser = argparse.ArgumentParser(description="Bitget 선물 자동매매 에이전트")
    parser.add_argument("--config", default="config/settings.yaml", help="설정 파일 경로")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    agent = TradingAgent(config)
    try:
        await agent.start()
    except KeyboardInterrupt:
        await agent.stop()


if __name__ == "__main__":
    # Python 3.11+ uvloop 사용 (있으면)
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

    asyncio.run(main())
