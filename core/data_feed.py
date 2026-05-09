"""실시간 데이터 피드 - WebSocket + REST 하이브리드"""
import asyncio
import json
import logging
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import websockets

from config import Config
from core.client import OKXClient

logger = logging.getLogger(__name__)

# OKX WebSocket 엔드포인트
# ※ 선물/스왑 캔들 채널은 반드시 /ws/v5/business 에서만 구독 가능
WS_PUBLIC   = "wss://ws.okx.com:8443/ws/v5/public"
WS_BUSINESS = "wss://ws.okx.com:8443/ws/v5/business"
WS_PRIVATE  = "wss://ws.okx.com:8443/ws/v5/private"

# REST 폴백: 캔들을 WS 없이 주기적으로 갱신하는 간격 (초)
_CANDLE_REST_REFRESH_SECS = 60

# ccxt timeframe → OKX WS channel 이름 매핑
_TF_TO_CHANNEL = {
    "1m": "candle1m",
    "3m": "candle3m",
    "5m": "candle5m",
    "15m": "candle15m",
    "30m": "candle30m",
    "1h": "candle1H",
    "2h": "candle2H",
    "4h": "candle4H",
    "6h": "candle6H",
    "12h": "candle12H",
    "1d": "candle1D",
    "1w": "candle1W",
}
# OKX WS channel → ccxt timeframe (역매핑)
_CHANNEL_TO_TF = {v: k for k, v in _TF_TO_CHANNEL.items()}


def _tf_to_okx_channel(tf: str) -> str:
    return _TF_TO_CHANNEL.get(tf.lower(), f"candle{tf}")


def _okx_channel_to_tf(channel: str) -> str:
    """candle15m → 15m, candle1H → 1h"""
    return _CHANNEL_TO_TF.get(channel, channel.replace("candle", ""))


class DataFeed:
    """실시간 시장 데이터 수집기"""

    def __init__(self, config: Config, client: OKXClient):
        self.config = config
        self.client = client

        # OKX instId 형식: BTC/USDT:USDT → BTC-USDT-SWAP
        raw_symbol = config.trading["symbol"]   # e.g. "BTC/USDT:USDT"
        base = raw_symbol.split("/")[0]          # "BTC"
        quote = raw_symbol.split("/")[1].split(":")[0]  # "USDT"
        self.symbol_ws = f"{base}-{quote}-SWAP"  # "BTC-USDT-SWAP"

        # 데이터 버퍼
        self.candles: dict[str, list] = {}   # {timeframe: [[ts, o, h, l, c, v], ...]}
        self.orderbook = {"bids": [], "asks": [], "ts": 0}
        self.trades: deque = deque(maxlen=2000)
        self.ticker: dict = {}
        self.funding_rate: float = 0.0
        self.oi: float = 0.0
        self.oi_prev: float = 0.0          # 1시간 전 OI (변화율 계산용)
        self.long_short_ratio: float = 0.0
        self._buy_vol_window: deque = deque(maxlen=2000)   # (ts_ms, size, side) 15분 집계용
        self._last_market_fetch: float = 0  # OI/롱숏 마지막 조회 시각

        # 콜백
        self._on_candle: Optional[Callable] = None
        self._on_tick: Optional[Callable] = None

        # 상태
        self._running = False
        self._ws = None

    # ── REST 데이터 초기 로드 ──────────────────────────
    def load_historical(self):
        """히스토리컬 캔들 데이터 초기 로드"""
        for tf in [
            self.config.strategy.get("timeframes", {}).get("primary", "15m"),
            self.config.strategy.get("timeframes", {}).get("confirm", "1h"),
            self.config.strategy.get("timeframes", {}).get("trend", "4h"),
        ]:
            raw = self.client.fetch_ohlcv(
                timeframe=tf,
                limit=self.config.data.get("candle_limit", 200),
            )
            self.candles[tf] = raw
            logger.info(f"📊 {tf} 캔들 {len(raw)}개 로드")

        # 펀딩비
        try:
            fr = self.client.fetch_funding_rate()
            self.funding_rate = float(fr.get("fundingRate", 0))
            logger.info(f"💰 펀딩비: {self.funding_rate:.6f}")
        except Exception as e:
            logger.warning(f"펀딩비 조회 실패: {e}")

    # ── 인디케이터 계산용 헬퍼 ─────────────────────────
    def get_closes(self, timeframe: str = "15m") -> np.ndarray:
        """종가 배열 반환"""
        candles = self.candles.get(timeframe, [])
        if not candles:
            return np.array([])
        return np.array([c[4] for c in candles], dtype=float)

    def get_highs(self, timeframe: str = "15m") -> np.ndarray:
        candles = self.candles.get(timeframe, [])
        return np.array([c[2] for c in candles], dtype=float)

    def get_lows(self, timeframe: str = "15m") -> np.ndarray:
        candles = self.candles.get(timeframe, [])
        return np.array([c[3] for c in candles], dtype=float)

    def get_volumes(self, timeframe: str = "15m") -> np.ndarray:
        candles = self.candles.get(timeframe, [])
        return np.array([c[5] for c in candles], dtype=float)

    def get_opens(self, timeframe: str = "15m") -> np.ndarray:
        candles = self.candles.get(timeframe, [])
        return np.array([c[1] for c in candles], dtype=float)

    def get_current_price(self) -> float:
        """현재 가격 — WebSocket ticker 우선, REST 폴백"""
        if self.ticker:
            price = float(self.ticker.get("last") or self.ticker.get("lastPr") or 0)
            if price > 0:
                return price
        # REST 폴백 (ccxt 표준화 키: last)
        t = self.client.fetch_ticker()
        return float(t.get("last", 0))

    # ── WebSocket ─────────────────────────────────────
    async def _ws_loop(self, endpoint: str, subscribe_msg: dict, label: str):
        """범용 WebSocket 루프 (자동 재연결 포함)"""
        while self._running:
            try:
                async with websockets.connect(
                    endpoint,
                    ping_interval=None,  # 앱 레벨 ping으로 처리
                ) as ws:
                    if label == "public":
                        self._ws = ws
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"🔌 OKX WS 연결 완료 ({label}) | {self.symbol_ws}")

                    alive = True

                    async def keepalive(ws=ws):
                        while alive:
                            await asyncio.sleep(25)
                            if alive:
                                try:
                                    await ws.send("ping")
                                except Exception:
                                    break

                    asyncio.ensure_future(keepalive())

                    async for msg in ws:
                        if msg == "pong":
                            continue
                        if msg == "ping":
                            await ws.send("pong")
                            continue
                        try:
                            data = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        if "event" in data:
                            ev = data.get("event")
                            if ev == "error":
                                logger.error(f"OKX WS 에러 ({label}): {data}")
                            elif ev == "subscribe":
                                ch = data.get("arg", {}).get("channel", "")
                                logger.info(f"✅ WS 구독 성공 ({label}): {ch}")
                            continue
                        await self._handle_message(data)

            except websockets.ConnectionClosed:
                alive = False
                logger.warning(f"⚠️ WS 연결 끊김 ({label}), 5초 후 재연결")
                await asyncio.sleep(5)
            except Exception as e:
                alive = False
                logger.error(f"WS 에러 ({label}): {e}")
                await asyncio.sleep(5)

    async def _subscribe_public(self):
        """퍼블릭 채널 구독 (호가·체결·틱 → public, 캔들 → business)
        ※ OKX 선물/스왑 캔들 채널은 /ws/v5/business 전용
        """
        tfs = self.config.strategy.get("timeframes", {})
        primary_tf = tfs.get("primary", "15m")
        confirm_tf = tfs.get("confirm", "1h")
        trend_tf   = tfs.get("trend",   "4h")

        # public 엔드포인트: tickers / trades / books
        public_msg = {
            "op": "subscribe",
            "args": [
                {"channel": "tickers", "instId": self.symbol_ws},
                {"channel": "trades",  "instId": self.symbol_ws},
                {"channel": "books5",  "instId": self.symbol_ws},
            ],
        }

        # business 엔드포인트: 캔들 (선물/스왑 전용)
        business_msg = {
            "op": "subscribe",
            "args": [
                {"channel": _tf_to_okx_channel(primary_tf), "instId": self.symbol_ws},
                {"channel": _tf_to_okx_channel(confirm_tf), "instId": self.symbol_ws},
                {"channel": _tf_to_okx_channel(trend_tf),   "instId": self.symbol_ws},
            ],
        }

        # 두 WS 루프를 동시에 실행
        await asyncio.gather(
            self._ws_loop(WS_PUBLIC,   public_msg,   "public"),
            self._ws_loop(WS_BUSINESS, business_msg, "business"),
            self._rest_candle_refresh(),
        )

    async def _rest_candle_refresh(self):
        """WS 보완용: 주기적으로 REST로 캔들 갱신 (WS 누락 방지)"""
        tfs = self.config.strategy.get("timeframes", {})
        tf_list = [
            tfs.get("primary", "15m"),
            tfs.get("confirm", "1h"),
            tfs.get("trend", "4h"),
        ]
        await asyncio.sleep(_CANDLE_REST_REFRESH_SECS)  # 최초 로드 후 대기
        while self._running:
            try:
                for tf in tf_list:
                    raw = self.client.fetch_ohlcv(
                        timeframe=tf,
                        limit=self.config.data.get("candle_limit", 200),
                    )
                    if raw:
                        self.candles[tf] = raw
                logger.debug(f"🔄 REST 캔들 갱신 완료 ({', '.join(tf_list)})")
            except Exception as e:
                logger.warning(f"REST 캔들 갱신 실패: {e}")
            await asyncio.sleep(_CANDLE_REST_REFRESH_SECS)

    async def _handle_message(self, data: dict):
        """수신 메시지 라우팅"""
        arg = data.get("arg", {})
        channel = arg.get("channel", "")

        if channel == "tickers":
            self._update_ticker(data.get("data", []))
        elif channel == "trades":
            self._update_trades(data.get("data", []))
        elif channel.startswith("candle"):
            self._update_candle(data.get("data", []), channel)
        elif channel.startswith("books"):
            self._update_orderbook(data.get("data", []))

    def _update_ticker(self, data: list):
        if data:
            prev_empty = not self.ticker
            self.ticker = data[0]
            if prev_empty:
                logger.info(
                    f"📡 OKX ticker 최초 수신 | 키={list(self.ticker.keys())} | "
                    f"last={self.ticker.get('last')}"
                )
            if self._on_tick:
                self._on_tick(self.ticker)

    def _update_trades(self, data: list):
        """OKX trade 데이터: {tradeId, instId, px, sz, side, ts}"""
        now_ms = time.time() * 1000
        for trade in data:
            self.trades.append(trade)
            try:
                ts = float(trade.get("ts", now_ms))
                size = float(trade.get("sz", 0))      # OKX: sz (size)
                side = trade.get("side", "")           # "buy" | "sell"
                self._buy_vol_window.append((ts, size, side))
            except Exception:
                pass

    def _update_candle(self, data: list, channel: str):
        """
        OKX 캔들 데이터: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        confirm="1" → 캔들 확정 (마감)
        """
        if not data:
            return
        tf = _okx_channel_to_tf(channel)
        if tf not in self.candles:
            self.candles[tf] = []

        for candle in data:
            ts = int(candle[0])
            ohlcv = [ts, float(candle[1]), float(candle[2]),
                     float(candle[3]), float(candle[4]), float(candle[5])]
            confirmed = len(candle) > 8 and candle[8] == "1"

            if self.candles[tf] and self.candles[tf][-1][0] == ts:
                self.candles[tf][-1] = ohlcv   # 기존 캔들 갱신
            else:
                self.candles[tf].append(ohlcv)  # 새 캔들
                max_len = self.config.data.get("candle_limit", 200)
                if len(self.candles[tf]) > max_len:
                    self.candles[tf] = self.candles[tf][-max_len:]

                if confirmed and self._on_candle:
                    self._on_candle(tf, ohlcv)

    def _update_orderbook(self, data: list):
        """OKX books5: bids/asks = [[price, qty, 0, numOrders], ...]"""
        if data:
            book = data[0]
            self.orderbook = {
                "bids": [[float(b[0]), float(b[1])] for b in book.get("bids", [])],
                "asks": [[float(a[0]), float(a[1])] for a in book.get("asks", [])],
                "ts": int(book.get("ts", time.time() * 1000)),
            }

    # ── 거래량 / OI / 롱숏 헬퍼 ─────────────────────────
    def get_volume_stats(self, timeframe: str = "15m") -> dict:
        volumes = self.get_volumes(timeframe)
        if len(volumes) < 20:
            return {}

        vol_current = float(volumes[-1])
        vol_sma20 = float(volumes[-20:].mean())
        vol_ratio = vol_current / vol_sma20 if vol_sma20 > 0 else 0

        vol_5 = volumes[-6:-1]
        if len(vol_5) >= 2 and vol_5[0] > 0:
            vol_trend_5 = (float(vol_5[-1]) - float(vol_5[0])) / float(vol_5[0]) * 100
        else:
            vol_trend_5 = 0.0

        now_ms = time.time() * 1000
        cutoff = now_ms - 15 * 60 * 1000
        buy_vol = sum(s for ts, s, side in self._buy_vol_window if ts >= cutoff and side == "buy")
        sell_vol = sum(s for ts, s, side in self._buy_vol_window if ts >= cutoff and side == "sell")
        total_vol = buy_vol + sell_vol
        buy_vol_pct = buy_vol / total_vol if total_vol > 0 else 0.5

        return {
            "vol_current": vol_current,
            "vol_sma20": vol_sma20,
            "vol_ratio": round(vol_ratio, 3),
            "vol_spike": vol_ratio > 2.0,
            "vol_dry": vol_ratio < 0.5,
            "vol_trend_5": round(vol_trend_5, 2),
            "buy_vol_pct": round(buy_vol_pct, 4),
        }

    def get_vol_delta(self, timeframe: str = "15m") -> dict:
        now_ms = time.time() * 1000
        candle_ms = 15 * 60 * 1000
        deltas = []
        for i in range(20):
            start = now_ms - (20 - i) * candle_ms
            end = start + candle_ms
            b = sum(s for ts, s, side in self._buy_vol_window if start <= ts < end and side == "buy")
            s_ = sum(s for ts, s, side in self._buy_vol_window if start <= ts < end and side == "sell")
            deltas.append(b - s_)

        cumulative = sum(deltas)
        slope_5 = (deltas[-1] - deltas[-6]) / 5 if len(deltas) >= 6 else 0

        return {
            "vol_delta": round(deltas[-1], 4) if deltas else 0,
            "vol_delta_cumulative": round(cumulative, 4),
            "cvd_slope_5": round(slope_5, 4),
        }

    def fetch_market_context(self):
        """OI / 롱숏 비율 REST 조회 (1분마다 호출)"""
        now = time.time()
        if now - self._last_market_fetch < 60:
            return
        self._last_market_fetch = now

        # OI 조회
        try:
            symbol = self.config.trading["symbol"]
            oi_data = self.client.exchange.fetch_open_interest(symbol)
            new_oi = float(
                oi_data.get("openInterestAmount")
                or oi_data.get("openInterest")
                or 0
            )
            if self.oi > 0:
                self.oi_prev = self.oi
            self.oi = new_oi
        except Exception as e:
            logger.debug(f"OI 조회 실패: {e}")

        # 롱숏 비율 조회 (OKX)
        try:
            resp = self.client.exchange.publicGetRubikStatContractsLongShortAccountRatio({
                "instId": self.symbol_ws,
                "period": "5m",
            })
            data = resp.get("data", [{}])
            if data:
                self.long_short_ratio = float(
                    data[0].get("longShortRatio")
                    or data[0].get("longAccountRatio")
                    or 0
                )
        except Exception as e:
            logger.debug(f"롱숏 비율 조회 실패: {e}")

    def get_oi_change_pct(self) -> float:
        """OI 변화율 % (1시간 기준)"""
        if self.oi_prev > 0:
            return (self.oi - self.oi_prev) / self.oi_prev * 100
        return 0.0

    # ── 이벤트 핸들러 등록 ────────────────────────────
    def on_candle_close(self, callback: Callable):
        self._on_candle = callback

    def on_tick(self, callback: Callable):
        self._on_tick = callback

    # ── 시작/종료 ─────────────────────────────────────
    async def start(self):
        self._running = True
        self.load_historical()
        await self._subscribe_public()

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("🔌 WebSocket 연결 종료")
