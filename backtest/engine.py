"""
백테스트 엔진 - 히스토리컬 데이터 기반 전략 검증

사용법:
    python -m backtest.engine --days 30 --leverage 10
    python -m backtest.engine --days 30 --export data/result.json
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from strategy.base import atr as calc_atr
from strategy.aggregator import STRATEGY_MAP

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    idx: int = 0
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_rr_achieved: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    drawdown_curve: list = field(default_factory=list)
    prices: list = field(default_factory=list)

    def to_json(self) -> dict:
        step = max(1, len(self.equity_curve) // 500)
        return {
            "metrics": {
                "total_trades": self.total_trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": round(self.win_rate, 4),
                "total_return_pct": round(self.total_return_pct, 4),
                "max_drawdown_pct": round(self.max_drawdown_pct, 4),
                "profit_factor": round(min(self.profit_factor, 999), 2),
                "avg_win_pct": round(self.avg_win_pct, 4),
                "avg_loss_pct": round(self.avg_loss_pct, 4),
                "sharpe_ratio": round(self.sharpe_ratio, 2),
                "avg_rr_achieved": round(self.avg_rr_achieved, 2),
            },
            "equity_curve": [round(e, 2) for e in self.equity_curve[::step]],
            "drawdown_curve": [round(d, 4) for d in self.drawdown_curve[::step]],
            "prices": [round(p, 2) for p in self.prices[::step]],
            "trades": [
                {
                    "idx": t.idx, "side": t.side,
                    "entry": round(t.entry_price, 2),
                    "exit": round(t.exit_price, 2),
                    "sl": round(t.stop_loss, 2),
                    "tp": round(t.take_profit, 2),
                    "pnl": round(t.pnl_pct, 4),
                    "reason": t.exit_reason,
                }
                for t in self.trades
            ],
        }

    def export_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        logger.info(f"결과 저장: {path}")


class BacktestEngine:
    def __init__(self, config, leverage=10, initial_balance=1000.0):
        self.config = config
        self.leverage = leverage
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.strategies = []
        for sc in config.strategy.get("active", []):
            cls = STRATEGY_MAP.get(sc["name"])
            if cls:
                self.strategies.append((cls(sc.get("params", {})), sc["weight"]))
        self.signal_threshold = config.strategy.get("signal_threshold", 0.6)
        self.risk_per_trade = config.risk.get("risk_per_trade_pct", 0.02)
        self.rr_ratio = config.risk.get("take_profit_rr_ratio", 3.0)
        self.sl_atr_mult = config.risk.get("stop_loss_atr_multiplier", 1.5)

    def run(self, opens, highs, lows, closes, volumes) -> BacktestResult:
        n = len(closes)
        if n < 50:
            return BacktestResult()
        trades, equity = [], [self.initial_balance]
        self.balance = self.initial_balance
        in_pos = False
        ps = pe = psl = ptp = pa = 0
        lb = 30
        for i in range(lb, n):
            ch, cl, cc = highs[i], lows[i], closes[i]
            if in_pos:
                hsl = (cl <= psl) if ps == "long" else (ch >= psl)
                htp = (ch >= ptp) if ps == "long" else (cl <= ptp)
                if hsl or htp:
                    ep, rsn = (psl, "sl") if hsl else (ptp, "tp")
                    rp = (ep - pe) / pe if ps == "long" else (pe - ep) / pe
                    lp = rp * self.leverage * pa
                    self.balance *= (1 + lp)
                    trades.append(BacktestTrade(i, ps, pe, ep, psl, ptp, lp, rsn))
                    in_pos = False
                    if self.balance <= self.initial_balance * 0.05:
                        equity.append(self.balance)
                        break
                equity.append(self.balance)
                continue
            hc, hh, hl, hv = closes[:i+1], highs[:i+1], lows[:i+1], volumes[:i+1]
            ws, tw, bsl, btp = 0.0, 0.0, 0.0, 0.0
            for st, w in self.strategies:
                if st.name == "scalping":
                    continue
                r = st.analyze(hc, hh, hl, hv, cc, rr_ratio=self.rr_ratio)
                ws += r.signal.value * r.confidence * w
                tw += w
                if r.stop_loss and r.take_profit:
                    bsl, btp = r.stop_loss, r.take_profit
            if tw == 0:
                equity.append(self.balance)
                continue
            sc = ws / tw
            if abs(sc) < self.signal_threshold:
                equity.append(self.balance)
                continue
            atr_v = calc_atr(hh, hl, hc, 14)[-1] if len(hc) >= 14 else cc * 0.015
            sld = atr_v * self.sl_atr_mult
            if sc > 0:
                ps, pe = "long", cc
                psl = max(bsl, cc - sld) if bsl else cc - sld
                ptp = btp if btp else cc + sld * self.rr_ratio
            else:
                ps, pe = "short", cc
                psl = min(bsl, cc + sld) if bsl else cc + sld
                ptp = btp if btp else cc - sld * self.rr_ratio
            sp = abs(pe - psl) / pe
            pa = min(self.risk_per_trade / (sp * self.leverage), 0.3) if sp > 0 else 0.01
            in_pos = True
            equity.append(self.balance)
        return self._compile(trades, equity, closes[lb:lb+len(equity)])

    def _compile(self, trades, equity_curve, prices):
        if not trades:
            return BacktestResult()
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        aw = float(np.mean([t.pnl_pct for t in wins])) if wins else 0
        al = float(np.mean([t.pnl_pct for t in losses])) if losses else 0
        gp = sum(t.pnl_pct for t in wins)
        gl = abs(sum(t.pnl_pct for t in losses))
        eq = np.array(equity_curve)
        pk = np.maximum.accumulate(eq)
        dd = (eq - pk) / pk
        rt = np.diff(eq) / eq[:-1]
        sh = float(np.mean(rt) / np.std(rt) * np.sqrt(365)) if len(rt) > 1 and np.std(rt) > 0 else 0.0
        return BacktestResult(
            total_trades=len(trades), wins=len(wins), losses=len(losses),
            win_rate=len(wins)/len(trades),
            total_return_pct=(self.balance - self.initial_balance) / self.initial_balance,
            max_drawdown_pct=float(np.min(dd)),
            profit_factor=gp/gl if gl > 0 else float("inf"),
            avg_win_pct=aw, avg_loss_pct=al, sharpe_ratio=sh,
            avg_rr_achieved=abs(aw/al) if al else 0,
            trades=trades, equity_curve=equity_curve,
            drawdown_curve=dd.tolist(),
            prices=prices.tolist() if hasattr(prices, 'tolist') else list(prices),
        )


def print_report(r):
    print("\n" + "=" * 60)
    print(f"  총 거래: {r.total_trades}  |  승률: {r.win_rate:.1%}  |  PF: {r.profit_factor:.2f}")
    print(f"  수익률: {r.total_return_pct:+.2%}  |  MDD: {r.max_drawdown_pct:.2%}  |  Sharpe: {r.sharpe_ratio:.2f}")
    print(f"  평균W: {r.avg_win_pct:+.2%}  |  평균L: {r.avg_loss_pct:+.2%}  |  R:R: 1:{r.avg_rr_achieved:.1f}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/settings.yaml")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--leverage", type=int, default=10)
    p.add_argument("--balance", type=float, default=1000.0)
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--export", default=None)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(a.config)
    from core.client import BitgetClient
    client = BitgetClient(cfg)
    tf_min = {"1m":1,"5m":5,"15m":15,"1h":60,"4h":240}
    limit = min(a.days * (1440 // tf_min.get(a.timeframe, 15)), 1000)
    raw = client.fetch_ohlcv(timeframe=a.timeframe, limit=limit)
    if not raw:
        return
    d = np.array(raw, dtype=float)
    eng = BacktestEngine(cfg, leverage=a.leverage, initial_balance=a.balance)
    res = eng.run(d[:,1], d[:,2], d[:,3], d[:,4], d[:,5])
    print_report(res)
    if a.export:
        res.export_json(a.export)

if __name__ == "__main__":
    main()
