"""텔레그램 알림 모듈"""
import logging
import asyncio
from typing import Optional

import httpx

from config import Config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """텔레그램 봇을 통한 트레이딩 알림"""

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, config: Config):
        self.enabled = config.telegram.get("enabled", False)
        self.token = config.telegram.get("bot_token", "")
        self.chat_id = config.telegram.get("chat_id", "")
        self.notify_on = config.telegram.get("notify_on", [])

        if self.enabled and (not self.token or self.token == "YOUR_TELEGRAM_BOT_TOKEN"):
            logger.warning("⚠️ 텔레그램 봇 토큰 미설정 - 알림 비활성화")
            self.enabled = False

    async def send(self, message: str, parse_mode: str = "HTML"):
        """비동기 메시지 전송"""
        if not self.enabled:
            return

        url = self.API_URL.format(token=self.token)
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=10)
                if resp.status_code != 200:
                    logger.error(f"텔레그램 전송 실패: {resp.text}")
        except Exception as e:
            logger.error(f"텔레그램 에러: {e}")

    def send_sync(self, message: str):
        """동기 메시지 전송 (이벤트루프 밖에서 사용)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send(message))
            else:
                loop.run_until_complete(self.send(message))
        except RuntimeError:
            asyncio.run(self.send(message))

    # ── 프리셋 메시지 ─────────────────────────────────

    async def notify_entry(
        self, side: str, amount: float, price: float,
        sl: float, tp: float, reasons: list = None,
        score: float = 0.0, leverage: int = 0,
        div_grade: str = "none", div_tf_count: int = 0,
        rr_ratio: float = 3.0,
    ):
        if "entry" not in self.notify_on:
            return
        direction = "LONG 📈" if side == "buy" else "SHORT 📉"
        sl_pct = abs(price - sl) / price * 100
        tp_pct = abs(tp - price) / price * 100
        rr_actual = tp_pct / sl_pct if sl_pct > 0 else rr_ratio

        # 등급 이모지
        grade_emoji = {"A": "🏆", "B": "🥇", "C": "🥈", "D": "🥉", "E": "⚠️"}.get(div_grade, "❔")
        grade_line = f"{grade_emoji} 다이버전스 등급: <b>{div_grade}급</b> ({div_tf_count}개 TF 확인)\n" if div_grade != "none" else ""

        reason_text = "\n".join(f"  • {r}" for r in (reasons or [])) or "  • 없음"
        msg = (
            f"{'🟢' if side == 'buy' else '🔴'} <b>포지션 진입 — {direction}</b>\n"
            f"\n"
            f"진입가:  <b>${price:,.2f}</b>\n"
            f"수량:    {amount} BTC  ({leverage}x 레버리지)\n"
            f"손절:    ${sl:,.2f}  (-{sl_pct:.2f}%)\n"
            f"익절:    ${tp:,.2f}  (+{tp_pct:.2f}%)  → R:R 1:{rr_actual:.1f}\n"
            f"신호점수: {score:+.3f}\n"
            f"\n"
            f"{grade_line}"
            f"<b>진입 근거</b>\n{reason_text}"
        )
        await self.send(msg)

    async def notify_exit(
        self, side: str, entry_price: float, exit_price: float,
        pnl_pct: float, pnl_usdt: float, reason: str,
        duration_min: float = 0,
        cumulative_stats: dict = None,
        balance: float = 0.0,
        leverage: int = 0,
        mfe_r: float = 0.0,
    ):
        if "exit" not in self.notify_on and "stop_loss" not in self.notify_on:
            return
        emoji = "✅" if pnl_usdt >= 0 else "❌"
        reason_map = {
            "emergency_sl": "🚨 비상 손절",
            "trailing_stop": "🔄 트레일링 스탑",
            "sl_server": "🛑 서버 손절",
            "tp_server": "🎯 서버 익절",
            "manual": "🖐 수동 청산",
            "drawdown_guard": "🛡 드로다운 가드",
        }
        reason_str = reason_map.get(reason, reason)
        direction = "LONG" if side == "buy" else "SHORT"

        # 잔고 대비 수익률
        balance_pct = pnl_usdt / balance if balance > 0 else 0.0
        lev_str = f"  ({leverage}x 레버리지: {pnl_pct:+.2%})" if leverage > 0 else ""
        pnl_line = (
            f"손익:    <b>{pnl_usdt:+.2f} USDT</b>  ({balance_pct:+.2%}){lev_str}"
        )

        # MFE R배수 (손익비 최적화 핵심)
        mfe_line = ""
        if mfe_r > 0:
            mfe_line = f"최대도달: <b>{mfe_r:.2f}R</b>  (이 트레이드의 최대 유리 이동)\n"

        # 누적 성과
        if cumulative_stats and cumulative_stats.get("total", 0) > 0:
            cs = cumulative_stats
            pf = cs.get("profit_factor", 0)
            pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
            stats_line = (
                f"\n<b>── 누적 성과 ──</b>\n"
                f"승률: <b>{cs['win_rate']:.1%}</b>  "
                f"({cs['wins']}승 {cs['losses']}패 / {cs['total']}회)\n"
                f"손익합: {cs['total_pnl_usdt']:+.2f} USDT  PF: {pf_str}"
            )
        else:
            stats_line = ""

        msg = (
            f"{emoji} <b>포지션 청산 — {direction}</b>\n"
            f"\n"
            f"진입가: ${entry_price:,.2f}\n"
            f"청산가: <b>${exit_price:,.2f}</b>\n"
            f"{pnl_line}\n"
            f"{mfe_line}"
            f"보유:   {duration_min:.0f}분\n"
            f"사유:   {reason_str}"
            f"{stats_line}"
        )
        await self.send(msg)

    async def notify_error(self, error: str):
        if "error" not in self.notify_on:
            return
        msg = f"🚨 <b>에러 발생</b>\n{error}"
        await self.send(msg)

    async def notify_indicators(
        self,
        price: float,
        ema_fast: float,
        ema_slow: float,
        hist_curr: float,
        hist_prev: float,
        rsi_val: float,
        last_signal=None,   # AggregatedSignal 객체
        signal_hint: str = "",  # 하위호환용 (미사용)
    ):
        """15분마다 지표 현황 — 다이버전스 트리거 기반"""
        if "indicators" not in self.notify_on:
            return

        # ── RSI 상태 ──
        if rsi_val >= 70:
            rsi_emoji, rsi_text = "🔥", f"{rsi_val:.0f} 과매수"
        elif rsi_val <= 30:
            rsi_emoji, rsi_text = "🧊", f"{rsi_val:.0f} 과매도"
        elif rsi_val >= 55:
            rsi_emoji, rsi_text = "💪", f"{rsi_val:.0f} 강세"
        elif rsi_val <= 45:
            rsi_emoji, rsi_text = "😰", f"{rsi_val:.0f} 약세"
        else:
            rsi_emoji, rsi_text = "😐", f"{rsi_val:.0f} 중립"

        ema_arrow = "📈" if ema_fast > ema_slow else "📉"
        macd_arrow = "▲" if hist_curr > hist_prev else "▼"
        macd_sign = "+" if hist_curr >= 0 else ""

        # ── 다이버전스 상태 (실제 진입 트리거) ──
        if last_signal is not None:
            sig = last_signal
            direction = sig.direction

            # TF별 다이버전스 상세 (extra에서 가져옴)
            # aggregator에서 div 정보 직접 사용
            grade = sig.div_grade
            tf_count = sig.div_tf_count
            score = sig.score
            cvd = sig.cvd_confirmed

            if direction == "long":
                dir_emoji = "🟢"
                dir_text = f"<b>롱 시그널 감지!</b> ({grade}급, {tf_count}개 TF)"
                score_bar = "▓" * min(int(abs(score) * 10), 10)
                div_block = (
                    f"\n🎯 <b>다이버전스 상태</b>\n"
                    f"   방향: {dir_emoji} {dir_text}\n"
                    f"   점수: {score:+.3f}  [{score_bar}]\n"
                    f"   CVD:  {'✅ 매수우위' if cvd else '❌ 미확인'}\n"
                )
            elif direction == "short":
                dir_emoji = "🔴"
                dir_text = f"<b>숏 시그널 감지!</b> ({grade}급, {tf_count}개 TF)"
                score_bar = "▓" * min(int(abs(score) * 10), 10)
                div_block = (
                    f"\n🎯 <b>다이버전스 상태</b>\n"
                    f"   방향: {dir_emoji} {dir_text}\n"
                    f"   점수: {score:+.3f}  [{score_bar}]\n"
                    f"   CVD:  {'✅ 매도우위' if cvd else '❌ 미확인'}\n"
                )
            else:
                # NEUTRAL — TF별 다이버전스 상세 표시
                reasons_summary = sig.reasons[0] if sig.reasons else "없음"
                # score_trend/score_breakout 으로 어느 방향 가까운지
                closer = ""
                if abs(sig.score_trend) > 0 or abs(sig.score_breakout) > 0:
                    net = sig.score_trend + sig.score_breakout
                    closer = f"  (점수 {net:+.3f}, 임계값 ±0.32)"
                div_block = (
                    f"\n⏳ <b>다이버전스 대기 중</b>\n"
                    f"   현재 점수: {score:+.3f}{closer}\n"
                    f"   상태: {reasons_summary[:60]}\n"
                )
        else:
            div_block = "\n⏳ <b>다이버전스</b>: 평가 대기 중\n"

        msg = (
            f"📊 <b>15분 지표 현황</b>\n"
            f"💰 BTC: <b>${price:,.1f}</b>\n"
            f"\n"
            f"{ema_arrow} EMA:  {ema_fast:,.0f} / {ema_slow:,.0f}  "
            f"({'정배열' if ema_fast > ema_slow else '역배열'})\n"
            f"{macd_arrow} MACD: {macd_sign}{hist_curr:.2f}  (전봉 {hist_prev:+.2f})\n"
            f"{rsi_emoji} RSI:  {rsi_text}"
            f"{div_block}"
        )
        await self.send(msg)

    async def notify_daily_summary(self, stats: dict, balance: float = 0.0):
        if "daily_summary" not in self.notify_on:
            return
        total = stats.get("total", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        win_rate = stats.get("win_rate", 0)
        pnl_usdt = stats.get("total_pnl_usdt", 0)
        pf = stats.get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        emoji = "✅" if pnl_usdt >= 0 else "❌"

        bal_line = f"잔고:   <b>${balance:,.2f} USDT</b>\n" if balance > 0 else ""
        trade_line = f"거래:   {total}회  ({wins}승 {losses}패)\n" if total > 0 else "거래:   0회\n"

        msg = (
            f"{emoji} <b>일간 요약</b>\n"
            f"\n"
            f"{bal_line}"
            f"{trade_line}"
            f"승률:   <b>{win_rate:.1%}</b>\n"
            f"손익:   <b>{pnl_usdt:+.2f} USDT</b>\n"
            f"PF:     {pf_str}"
        )
        await self.send(msg)
