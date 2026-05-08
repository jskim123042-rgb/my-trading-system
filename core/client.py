"""OKX 거래소 클라이언트 래퍼 (ccxt 기반)"""
import ccxt
import logging
from typing import Optional
from config import Config

logger = logging.getLogger(__name__)


class OKXClient:
    """OKX V5 API 클라이언트 - ccxt swap 모드"""

    def __init__(self, config: Config):
        self.config = config
        self.exchange = ccxt.okx({
            "apiKey": config.exchange["api_key"],
            "secret": config.exchange["secret_key"],
            "password": config.exchange["passphrase"],
            "options": {
                "defaultType": "swap",
            },
            "enableRateLimit": True,
        })

        if config.exchange.get("sandbox", False):
            self.exchange.set_sandbox_mode(True)

        self._margin_mode = config.trading.get("margin_mode", "isolated")
        self._init_market()

    def _init_market(self):
        """마켓 로드 및 레버리지 초기화"""
        self.exchange.load_markets()
        symbol = self.config.trading["symbol"]
        leverage = self.config.leverage["default"]

        # 마진 모드 설정
        try:
            self.exchange.set_margin_mode(self._margin_mode, symbol)
        except Exception as e:
            logger.warning(f"마진 모드 설정 실패 (이미 설정됨?): {e}")

        # 레버리지 설정
        self.set_leverage(leverage)
        logger.info(
            f"✅ OKX 클라이언트 초기화 완료 | {symbol} | {leverage}x | "
            f"sandbox={self.config.exchange.get('sandbox')}"
        )

    # ── 계좌 ──────────────────────────────────────────
    def get_balance(self) -> dict:
        """USDT 선물 계좌 잔고 조회"""
        balance = self.exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return {
            "total": float(usdt.get("total", 0)),
            "free": float(usdt.get("free", 0)),
            "used": float(usdt.get("used", 0)),
        }

    def get_positions(self, symbol: Optional[str] = None) -> list:
        """현재 열린 포지션 조회"""
        positions = self.exchange.fetch_positions(
            symbols=[symbol or self.config.trading["symbol"]]
        )
        return [p for p in positions if float(p.get("contracts", 0)) > 0]

    # ── 레버리지 ──────────────────────────────────────
    def set_leverage(self, leverage: int):
        """레버리지 설정 (동적 조절용)
        OKX isolated margin: long/short 양쪽 모두 설정 필요 (hedge mode)
        one-way mode 폴백 지원
        """
        lev = max(
            self.config.leverage["min"],
            min(leverage, self.config.leverage["max"]),
        )
        symbol = self.config.trading["symbol"]
        mgn = self._margin_mode  # "isolated" | "cross"
        try:
            try:
                # hedge mode: posSide 별도 지정 필요
                self.exchange.set_leverage(lev, symbol, params={"mgnMode": mgn, "posSide": "long"})
                self.exchange.set_leverage(lev, symbol, params={"mgnMode": mgn, "posSide": "short"})
            except Exception:
                # one-way (net) mode: posSide 없이 한 번만
                self.exchange.set_leverage(lev, symbol, params={"mgnMode": mgn})
            logger.info(f"레버리지 → {lev}x ({mgn})")
        except Exception as e:
            logger.error(f"레버리지 설정 실패: {e}")

    # ── 단위 변환 헬퍼 ────────────────────────────────
    # OKX BTC-USDT-SWAP: 1계약 = 0.01 BTC
    # PositionSizer는 BTC 단위로 계산 → API 전달 시 계약 수로 변환 필요
    BTC_PER_CONTRACT = 0.01

    def _btc_to_contracts(self, btc_amount: float) -> int:
        """BTC 수량 → OKX 계약 수 (정수)
        amount_to_precision 미사용: OKX가 BTC로 오해석해 0.01배 축소되는 버그 방지
        """
        contracts = int(round(btc_amount / self.BTC_PER_CONTRACT))
        return max(1, contracts)

    def _contracts_to_btc(self, contracts: float) -> float:
        """OKX 계약 수 → BTC 수량"""
        return contracts * self.BTC_PER_CONTRACT

    # ── 주문 ──────────────────────────────────────────
    def market_open(
        self,
        side: str,
        amount: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> dict:
        """
        시장가 진입 후 TP/SL 알고 주문 별도 등록
        side: 'buy' (롱) | 'sell' (숏)
        amount: BTC 수량 (내부에서 계약 수로 변환)
        """
        symbol = self.config.trading["symbol"]
        contracts = self._btc_to_contracts(amount)
        logger.info(f"🔄 수량 변환 | {amount} BTC → {contracts} 계약")

        # 1단계: 순수 시장가 진입 (TP/SL 없이)
        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=contracts,
            params={"tdMode": self._margin_mode},
        )
        logger.info(
            f"📈 진입 | {side.upper()} {contracts}계약({amount}BTC) | "
            f"TP={tp_price} SL={sl_price} | ID={order['id']}"
        )

        # 2단계: TP/SL 알고 주문 별도 등록
        if tp_price or sl_price:
            try:
                close_side = "sell" if side == "buy" else "buy"
                algo_params: dict = {
                    "tdMode": self._margin_mode,
                    "reduceOnly": True,
                }
                if tp_price:
                    algo_params["tpTriggerPx"] = str(round(tp_price, 1))
                    algo_params["tpOrdPx"] = "-1"
                    algo_params["tpTriggerPxType"] = "mark"
                if sl_price:
                    algo_params["slTriggerPx"] = str(round(sl_price, 1))
                    algo_params["slOrdPx"] = "-1"
                    algo_params["slTriggerPxType"] = "mark"

                self.exchange.create_order(
                    symbol=symbol,
                    type="oco" if (tp_price and sl_price) else "conditional",
                    side=close_side,
                    amount=contracts,
                    params=algo_params,
                )
                logger.info(f"🛡️ TP/SL 등록 | TP={tp_price} SL={sl_price}")
            except Exception as e:
                logger.warning(f"TP/SL 별도 등록 실패 (소프트웨어 추적으로 관리): {e}")

        return order

    def market_close(self, side: str, amount: float) -> dict:
        """시장가 청산
        amount: BTC 수량 (내부에서 계약 수로 변환)
        """
        symbol = self.config.trading["symbol"]
        close_side = "sell" if side == "buy" else "buy"
        contracts = self._btc_to_contracts(amount)

        order = self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=close_side,
            amount=contracts,
            params={
                "tdMode": self._margin_mode,
                "reduceOnly": True,
            },
        )
        logger.info(f"📉 청산 | {close_side.upper()} {contracts}계약({amount}BTC) | ID={order['id']}")
        return order

    def set_trailing_stop(
        self,
        side: str,
        amount: float,
        trigger_price: float,
        callback_ratio: float = 1.5,
    ) -> dict:
        """트레일링 스탑 (클라이언트 사이드 사용 권장 - 서버 사이드 미사용)"""
        logger.info(f"🔄 트레일링 스탑 (클라이언트 사이드) | callback={callback_ratio}% | trigger={trigger_price}")
        return {}

    # ── 시장 데이터 ────────────────────────────────────
    def fetch_ohlcv(self, timeframe: str = "15m", limit: int = 200) -> list:
        """캔들 데이터 조회"""
        return self.exchange.fetch_ohlcv(
            self.config.trading["symbol"],
            timeframe=timeframe,
            limit=limit,
        )

    def fetch_orderbook(self, depth: int = 20) -> dict:
        """오더북 조회"""
        return self.exchange.fetch_order_book(
            self.config.trading["symbol"],
            limit=depth,
        )

    def fetch_ticker(self) -> dict:
        """현재 가격 정보"""
        return self.exchange.fetch_ticker(self.config.trading["symbol"])

    def fetch_funding_rate(self) -> dict:
        """펀딩비 조회"""
        return self.exchange.fetch_funding_rate(self.config.trading["symbol"])


# 하위 호환성 alias
BitgetClient = OKXClient
