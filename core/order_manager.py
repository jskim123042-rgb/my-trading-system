"""주문 실행 및 포지션 관리"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from config import Config
from core.client import BitgetClient

logger = logging.getLogger(__name__)


class Side(Enum):
    LONG = "buy"
    SHORT = "sell"


@dataclass
class Trade:
    """개별 트레이드 기록"""
    id: str = ""
    side: str = ""
    entry_price: float = 0.0
    amount: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    leverage: int = 8              # 진입 시 실제 사용한 레버리지
    trailing_active: bool = False
    high_water_mark: float = 0.0   # 롱 최고가 추적 (클라이언트 트레일링)
    low_water_mark: float = float("inf")  # 숏 최저가 추적
    entry_time: float = 0.0
    exit_price: float = 0.0
    exit_time: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    entry_reasons: list = field(default_factory=list)
    mfe_pct: float = 0.0   # Max Favorable Excursion (최대 유리 이동 %)
    mae_pct: float = 0.0   # Max Adverse Excursion (최대 불리 이동 %)
    mfe_r: float = 0.0     # MFE in R multiples (손절폭 대비 최대 수익 배수)


class OrderManager:
    """주문 실행 + 포지션 라이프사이클 관리"""

    def __init__(self, config: Config, client: BitgetClient):
        self.config = config
        self.client = client
        self.current_trade: Optional[Trade] = None
        self.trade_history: list[Trade] = []

    @property
    def has_position(self) -> bool:
        return self.current_trade is not None

    # ── 재시작 시 포지션 복구 ──────────────────────────
    def sync_position(self) -> bool:
        """
        거래소에서 현재 포지션을 읽어 current_trade에 복구.
        재시작 또는 연결 복구 시 호출.
        Returns: True if position was found and restored
        """
        try:
            positions = self.client.get_positions()
            if not positions:
                logger.info("🔍 복구 확인: 열린 포지션 없음")
                return False

            pos = positions[0]
            # ccxt 정규화: 'long'/'short', Bitget raw: 'buy'/'sell' 또는 'long'/'short' 모두 처리
            side_raw = (
                pos.get("side")
                or pos.get("info", {}).get("holdSide")
                or ""
            ).lower()
            logger.info(f"🔍 포지션 복구 raw: side={side_raw} | keys={list(pos.keys())}")
            side = "buy" if side_raw in ("long", "buy") else "sell"
            entry_price = float(
                pos.get("entryPrice")
                or pos.get("info", {}).get("openPriceAvg")
                or pos.get("info", {}).get("averageOpenPrice")
                or 0
            )
            # contracts: OKX 계약 수 → BTC 수량으로 변환 (1계약 = 0.01 BTC)
            contracts = float(
                pos.get("contracts")
                or pos.get("info", {}).get("total")
                or 0
            )
            amount = contracts * 0.01 if contracts > 0 else 0.0
            sl = float(pos.get("stopLossPrice") or 0)
            tp = float(pos.get("takeProfitPrice") or 0)

            if entry_price <= 0 or amount <= 0:
                logger.warning(f"⚠️ 포지션 복구 실패: entry_price={entry_price} amount={amount}")
                return False

            trade = Trade(
                id=pos.get("id", "restored"),
                side=side,
                entry_price=entry_price,
                amount=amount,
                stop_loss=sl,
                take_profit=tp,
                entry_time=time.time(),
                high_water_mark=entry_price,
                low_water_mark=entry_price,
                entry_reasons=["[복구됨] 재시작 전 포지션"],
            )
            self.current_trade = trade
            logger.warning(
                f"♻️ 포지션 복구 완료 | {side.upper()} {amount} @ {entry_price} | "
                f"SL={sl} TP={tp}"
            )
            return True

        except Exception as e:
            logger.error(f"포지션 복구 중 에러: {e}")
            return False

    # ── 진입 ──────────────────────────────────────────
    def open_position(
        self,
        side: Side,
        amount: float,
        stop_loss: float,
        take_profit: float,
        current_price: float = 0.0,
        entry_reasons: list = None,
        leverage: int = 0,
    ) -> Optional[Trade]:
        """
        포지션 진입 (시장가) + 서버 사이드 TP/SL 동시 설정
        """
        if self.has_position:
            logger.warning("⚠️ 이미 포지션 보유 중 - 진입 거부")
            return None

        try:
            order = self.client.market_open(
                side=side.value,
                amount=amount,
                tp_price=take_profit,
                sl_price=stop_loss,
            )

            # 체결가 확인 (fills → price → average → current_price 폴백 순)
            fills = order.get("trades", [])
            entry_price = (
                float(fills[0]["price"]) if fills
                else float(order.get("price") or order.get("average") or current_price or 0)
            )
            if entry_price <= 0:
                logger.warning("⚠️ 주문 응답에 체결가 없음 — 현재가로 대체")

            actual_leverage = leverage if leverage > 0 else self.config.leverage["default"]
            trade = Trade(
                id=order["id"],
                side=side.value,
                entry_price=entry_price,
                amount=amount,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=actual_leverage,
                entry_time=time.time(),
                high_water_mark=entry_price,
                low_water_mark=entry_price,
                entry_reasons=entry_reasons or [],
            )
            self.current_trade = trade

            logger.info(
                f"✅ 포지션 오픈 | {side.name} {amount} @ {entry_price} | "
                f"레버리지={actual_leverage}x | SL={stop_loss} TP={take_profit}"
            )
            return trade

        except Exception as e:
            logger.error(f"❌ 주문 실패: {e}")
            return None

    # ── 청산 ──────────────────────────────────────────
    def close_position(self, reason: str = "manual") -> Optional[Trade]:
        """현재 포지션 시장가 청산"""
        if not self.has_position:
            logger.warning("⚠️ 청산할 포지션 없음")
            return None

        trade = self.current_trade
        try:
            order = self.client.market_close(
                side=trade.side,
                amount=trade.amount,
            )

            # 체결가
            fills = order.get("trades", [])
            exit_price = (
                float(fills[0]["price"]) if fills
                else float(order.get("price") or order.get("average") or 0)
            )
            if exit_price <= 0:
                logger.warning("⚠️ 청산 응답에 체결가 없음 — PnL 계산 불가, 0으로 기록")

            # PnL 계산 (entry_price 또는 exit_price가 0이면 0으로 처리)
            if trade.entry_price > 0 and exit_price > 0:
                if trade.side == "buy":  # 롱
                    pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                else:  # 숏
                    pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
            else:
                pnl_pct = 0.0

            leverage = trade.leverage  # 진입 시 실제 사용한 레버리지
            pnl_pct_leveraged = pnl_pct * leverage
            nominal = trade.amount * trade.entry_price  # 명목가치 (USDT)

            trade.exit_price = exit_price
            trade.exit_time = time.time()
            trade.pnl_pct = pnl_pct_leveraged          # 레버리지 적용 수익률 (표시용)
            trade.pnl = pnl_pct * nominal               # 실제 USDT 손익 = 가격변화% × 명목가치
            trade.exit_reason = reason

            self.trade_history.append(trade)
            self.current_trade = None

            emoji = "🟢" if trade.pnl >= 0 else "🔴"
            logger.info(
                f"{emoji} 포지션 청산 | {reason} | {trade.side.upper()} @ {exit_price} | "
                f"PnL: {trade.pnl_pct:+.2%}"
            )
            return trade

        except Exception as e:
            logger.error(f"❌ 청산 실패: {e}")
            return None

    # ── 포지션 모니터링 (틱마다 호출) ──────────────────
    def check_position(self, current_price: float) -> Optional["Trade"]:
        """
        실시간 포지션 상태 체크
        - 서버 SL 외에 클라이언트 사이드 이중 안전장치
        - 수익 1R 달성 시 클라이언트 트레일링 스탑 전환
        - 30초마다 거래소 폴링 → 서버 SL/TP 체결 감지
        Returns: 청산된 Trade (청산 발생 시), 없으면 None
        """
        if not self.has_position:
            return None

        trade = self.current_trade

        if trade.entry_price <= 0:
            logger.warning("⚠️ entry_price=0, 체결가 미수신 — 모니터링 스킵")
            return None

        # ── 거래소 포지션 실제 확인 (30초마다) ──
        # 서버 사이드 SL/TP 체결 시 봇이 모르는 버그 방지
        now = time.time()
        if not hasattr(self, '_last_exchange_check'):
            self._last_exchange_check = 0
        if now - self._last_exchange_check >= 30:
            self._last_exchange_check = now
            try:
                positions = self.client.get_positions()
                if not positions:
                    # 거래소에 포지션 없음 = 서버 SL/TP 체결됨
                    logger.warning("⚠️ 거래소 포지션 없음 감지 — 서버 SL/TP 체결로 판단")
                    # 현재가 기준으로 손익 추정
                    if trade.side == "buy":
                        pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                    else:
                        pnl_pct = (trade.entry_price - current_price) / trade.entry_price
                    leverage = trade.leverage  # 진입 시 실제 레버리지
                    nominal = trade.amount * trade.entry_price
                    trade.exit_price = current_price
                    trade.exit_time = now
                    trade.pnl_pct = pnl_pct * leverage
                    trade.pnl = pnl_pct * nominal
                    # 방향으로 손절/익절 판단
                    if trade.side == "sell":
                        reason = "sl_server" if current_price > trade.entry_price else "tp_server"
                    else:
                        reason = "sl_server" if current_price < trade.entry_price else "tp_server"
                    trade.exit_reason = reason
                    emoji = "🟢" if trade.pnl >= 0 else "🔴"
                    logger.info(f"{emoji} 서버 SL/TP 체결 감지 | {reason} | PnL={trade.pnl_pct:+.2%}")
                    self.trade_history.append(trade)
                    self.current_trade = None
                    return trade
            except Exception as e:
                logger.warning(f"거래소 포지션 확인 실패: {e}")

        # 수익률 계산
        if trade.side == "buy":
            unrealized_pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            unrealized_pnl_pct = (trade.entry_price - current_price) / trade.entry_price

        leverage = trade.leverage  # 진입 시 실제 레버리지
        unrealized_leveraged = unrealized_pnl_pct * leverage

        # MFE/MAE 실시간 추적 (청산 후 SL/TP 최적화 분석용)
        if unrealized_pnl_pct > 0:
            trade.mfe_pct = max(trade.mfe_pct, unrealized_pnl_pct)
        else:
            trade.mae_pct = max(trade.mae_pct, abs(unrealized_pnl_pct))

        # R 배수 추적: 손절폭 대비 얼마나 유리하게 움직였는지
        # mfe_r = MFE(가격 이동) / SL 폭 → 1:X 손익비 최적화 핵심 지표
        sl_dist = abs(trade.entry_price - trade.stop_loss)
        if sl_dist > 0:
            if trade.side == "buy":
                current_r = (current_price - trade.entry_price) / sl_dist
            else:
                current_r = (trade.entry_price - current_price) / sl_dist
            if current_r > trade.mfe_r:
                trade.mfe_r = round(current_r, 3)

        # 1. 클라이언트 사이드 비상 손절
        sl_max = self.config.risk.get("stop_loss_max_pct", 0.03)
        if unrealized_leveraged < -sl_max * leverage:
            logger.warning(f"🚨 비상 손절 발동! PnL={unrealized_leveraged:+.2%}")
            return self.close_position(reason="emergency_sl")

        callback_pct = self.config.risk.get("trailing_stop_callback", 1.5) / 100
        risk_per_trade = abs(trade.entry_price - trade.stop_loss)

        # 2. 1R 수익 달성 시 트레일링 스탑 활성화
        if not trade.trailing_active:
            if abs(current_price - trade.entry_price) >= risk_per_trade:
                trade.trailing_active = True
                logger.info(f"🔄 트레일링 스탑 활성화 | callback={callback_pct:.1%} | PnL={unrealized_leveraged:+.2%}")

        # 3. 클라이언트 사이드 트레일링 스탑 체크
        if trade.trailing_active:
            if trade.side == "buy":
                trade.high_water_mark = max(trade.high_water_mark, current_price)
                trailing_sl = trade.high_water_mark * (1 - callback_pct)
                if current_price <= trailing_sl:
                    logger.info(f"🔄 트레일링 청산 | 고점={trade.high_water_mark:.1f} → SL={trailing_sl:.1f}")
                    return self.close_position(reason="trailing_stop")
            else:
                trade.low_water_mark = min(trade.low_water_mark, current_price)
                trailing_sl = trade.low_water_mark * (1 + callback_pct)
                if current_price >= trailing_sl:
                    logger.info(f"🔄 트레일링 청산 | 저점={trade.low_water_mark:.1f} → SL={trailing_sl:.1f}")
                    return self.close_position(reason="trailing_stop")

        return None

    # ── 통계 ──────────────────────────────────────────
    @staticmethod
    def _pnl(t) -> float:
        return t["pnl"] if isinstance(t, dict) else t.pnl

    @staticmethod
    def _pnl_pct(t) -> float:
        return t["pnl_pct"] if isinstance(t, dict) else t.pnl_pct

    def get_stats(self) -> dict:
        """트레이딩 통계 (Trade 객체 + dict 혼용 지원 — 재시작 복원용)"""
        if not self.trade_history:
            return {"total": 0, "win_rate": 0, "avg_pnl": 0,
                    "wins": 0, "losses": 0, "profit_factor": 0,
                    "total_pnl_usdt": 0, "total_pnl_pct": 0}

        wins   = [t for t in self.trade_history if self._pnl(t) > 0]
        losses = [t for t in self.trade_history if self._pnl(t) <= 0]

        avg_win  = sum(self._pnl_pct(t) for t in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(self._pnl_pct(t) for t in losses) / len(losses) if losses else 0

        return {
            "total":          len(self.trade_history),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       len(wins) / len(self.trade_history),
            "avg_win_pct":    avg_win,
            "avg_loss_pct":   avg_loss,
            "profit_factor":  abs(avg_win / avg_loss) if avg_loss else float("inf"),
            "total_pnl_pct":  sum(self._pnl_pct(t) for t in self.trade_history),
            "total_pnl_usdt": sum(self._pnl(t)     for t in self.trade_history),
        }
