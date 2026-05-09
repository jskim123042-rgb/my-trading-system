"""
모의 트레이더 (Paper Trader)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실거래에 영향 없이 Strategy A / B를 병렬로 실행하고
진입/청산 결과를 paper_trades 테이블에 기록한다.

호출 방법:
    paper_trader = PaperTrader(trade_logger)
    paper_trader.tick(data_feed, current_price)  # 메인 루프에서 매 틱 호출
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from strategy.paper_strategies import PaperStrategyA, PaperStrategyB, PaperSignal
from monitor.trade_logger import TradeLogger
from monitor.telegram_bot import TelegramNotifier

logger = logging.getLogger("paper_trader")


SLIPPAGE_PER_SIDE = 0.0005   # OKX 테이커 수수료 0.05% (진입+청산 = 0.10%)
FUNDING_INTERVAL_H = 8       # OKX 펀딩비 주기 (8시간)


@dataclass
class PaperPosition:
    row_id: int
    strategy: str       # "A" | "B"
    side: str           # "long" | "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    open_time: float    # time.time()
    balance_at_entry: float = 0.0
    funding_rate: float = 0.0   # 진입 시점 실제 funding rate


class PaperTrader:
    """Strategy A / B 모의 포지션 관리"""

    def __init__(self, trade_logger: TradeLogger, notifier: Optional[TelegramNotifier] = None):
        self.logger = trade_logger
        self.notifier = notifier
        self.strategy_a = PaperStrategyA()
        self.strategy_b = PaperStrategyB()
        # 각 전략당 하나의 모의 포지션만 유지
        self._pos_a: Optional[PaperPosition] = None
        self._pos_b: Optional[PaperPosition] = None
        self._restore_positions()

    # ── 재시작 복구 ───────────────────────────────────
    def _restore_positions(self):
        """봇 재시작 시 DB의 미청산 모의 포지션 복구"""
        open_positions = self.logger.load_open_paper_positions()
        if not open_positions:
            return
        for p in open_positions:
            # open_ts → Unix timestamp 변환
            try:
                dt = datetime.fromisoformat(p["open_ts"].replace("Z", "+00:00"))
                open_time = dt.timestamp()
            except Exception:
                open_time = time.time()

            pos = PaperPosition(
                row_id=p["row_id"],
                strategy=p["strategy"],
                side=p["side"],
                entry_price=p["entry_price"],
                stop_loss=p["stop_loss"],
                take_profit=p["take_profit"],
                open_time=open_time,
                balance_at_entry=p["balance_at_entry"],
                funding_rate=0.0,  # 재시작 시 알 수 없으므로 0으로 복구 (펀딩비 보수적 처리)
            )
            if p["strategy"] == "A" and self._pos_a is None:
                self._pos_a = pos
            elif p["strategy"] == "B" and self._pos_b is None:
                self._pos_b = pos
            logger.info(
                f"♻️ 모의 포지션 복구 [{p['strategy']}] {p['side'].upper()} "
                f"진입가={p['entry_price']:.1f} SL={p['stop_loss']:.1f} TP={p['take_profit']:.1f}"
            )

    # ── 매 틱 호출 ────────────────────────────────────
    def tick(self, data_feed, current_price: float, balance: float = 0.0):
        """메인 루프에서 5초마다 호출 — 진입 체크 + 청산 체크"""
        if current_price <= 0:
            return

        opens = data_feed.get_opens("15m")
        closes = data_feed.get_closes("15m")
        highs = data_feed.get_highs("15m")
        lows = data_feed.get_lows("15m")
        funding_rate = getattr(data_feed, "funding_rate", 0.0)

        if len(closes) < 40:
            return

        self._process_strategy(
            "A", self._pos_a, opens, closes, highs, lows, current_price, balance, funding_rate,
        )
        self._process_strategy(
            "B", self._pos_b, opens, closes, highs, lows, current_price, balance, funding_rate,
        )

    # ── 전략별 처리 ───────────────────────────────────
    def _process_strategy(
        self,
        name: str,
        pos: Optional[PaperPosition],
        opens, closes, highs, lows,
        current_price: float,
        balance: float = 0.0,
        funding_rate: float = 0.0,
    ):
        # 1) 보유 포지션 있으면 청산 체크
        if pos is not None:
            closed = self._check_exit(pos, current_price)
            if closed:
                if name == "A":
                    self._pos_a = None
                else:
                    self._pos_b = None
            return

        # 2) 포지션 없으면 진입 시그널 확인
        if name == "A":
            signal = self.strategy_a.analyze(opens, closes, highs, lows, current_price)
        else:
            signal = self.strategy_b.analyze(opens, closes, highs, lows, current_price)

        if signal.side == "neutral":
            return

        # 3) 진입 기록
        row_id = self.logger.log_paper_trade_open(
            strategy=name,
            side=signal.side,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_rsi=signal.rsi_val,
            entry_adx=signal.adx_val,
            entry_bb_pct_b=signal.bb_pct_b,
            reason=signal.reason,
            balance_at_entry=balance,
        )
        if row_id < 0:
            return

        new_pos = PaperPosition(
            row_id=row_id,
            strategy=name,
            side=signal.side,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            open_time=time.time(),
            balance_at_entry=balance,
            funding_rate=funding_rate,
        )
        if name == "A":
            self._pos_a = new_pos
        else:
            self._pos_b = new_pos

        logger.info(
            f"📝 모의진입 [{name}] {signal.side.upper()} "
            f"진입={signal.entry_price:.1f} SL={signal.stop_loss:.1f} TP={signal.take_profit:.1f} "
            f"| {signal.reason}"
        )
        if self.notifier:
            import asyncio
            asyncio.ensure_future(self.notifier.notify_paper_entry(
                strategy=name, side=signal.side,
                entry=signal.entry_price, sl=signal.stop_loss, tp=signal.take_profit,
                reason=signal.reason,
            ))

    # ── 청산 체크 ─────────────────────────────────────
    def _check_exit(self, pos: PaperPosition, current_price: float) -> bool:
        """SL/TP 도달 여부 확인. 청산되면 True 반환."""
        if pos.side == "long":
            hit_sl = current_price <= pos.stop_loss
            hit_tp = current_price >= pos.take_profit
        else:
            hit_sl = current_price >= pos.stop_loss
            hit_tp = current_price <= pos.take_profit

        if not (hit_sl or hit_tp):
            return False

        outcome = "tp" if hit_tp else "sl"
        exit_price = pos.take_profit if hit_tp else pos.stop_loss

        if pos.side == "long":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        duration_min = (time.time() - pos.open_time) / 60

        # 슬리피지: 진입 + 청산 각 0.05% = 총 0.10%
        slippage_pct = SLIPPAGE_PER_SIDE * 2

        # 펀딩비: 실제 funding_rate × 보유한 8시간 주기 수
        # 롱이면 양수 funding_rate = 비용, 숏이면 반대
        funding_periods = (duration_min / 60) / FUNDING_INTERVAL_H
        if pos.side == "long":
            funding_pct = pos.funding_rate * funding_periods
        else:
            funding_pct = -pos.funding_rate * funding_periods  # 숏은 펀딩비 수취
        funding_pct = max(funding_pct, 0.0)  # 수취는 0으로 처리 (보수적 계산)

        self.logger.log_paper_trade_close(
            row_id=pos.row_id,
            exit_price=exit_price,
            pnl_pct=round(pnl_pct, 6),
            exit_reason=outcome,
            duration_min=round(duration_min, 2),
            balance_at_entry=pos.balance_at_entry,
            slippage_pct=round(slippage_pct, 6),
            funding_pct=round(funding_pct, 6),
        )

        net_pnl = pnl_pct - slippage_pct - funding_pct
        sim_pnl_usdt = round(pos.balance_at_entry * net_pnl, 4) if pos.balance_at_entry > 0 else 0.0
        logger.info(
            f"📝 모의청산 [{pos.strategy}] {pos.side.upper()} "
            f"{'✅TP' if outcome == 'tp' else '❌SL'} "
            f"손익={pnl_pct:+.2%} 슬리피지={slippage_pct:.2%} "
            f"펀딩={funding_pct:.3%} 실질={net_pnl:+.2%} 보유={duration_min:.0f}분"
        )
        if self.notifier:
            import asyncio
            asyncio.ensure_future(self.notifier.notify_paper_exit(
                strategy=pos.strategy, side=pos.side,
                entry=pos.entry_price, exit_price=exit_price,
                pnl_pct=pnl_pct, net_pnl_pct=net_pnl,
                outcome=outcome, duration_min=duration_min,
                sim_pnl_usdt=sim_pnl_usdt,
            ))
        return True
