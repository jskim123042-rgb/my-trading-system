"""
그리드 서치 파라미터 최적화 엔진 (고속)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용법:
  python -m backtest.optimizer --strategy trend_follow --quick --demo
  python -m backtest.optimizer --strategy risk --demo --walk-forward 0.7
  python -m backtest.optimizer --strategy trend_follow --demo --export data/optim.json
"""
import argparse
import copy
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.fast_backtest import (
    fast_trend_follow, fast_breakout, fast_combined,
    FastResult, FAST_STRATEGIES,
)
from backtest.param_grids import (
    STRATEGY_GRIDS, RISK_GRID, QUICK_GRIDS, RISK_QUICK,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimResult:
    params: dict = field(default_factory=dict)
    total_trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_rr_achieved: float = 0.0
    score: float = 0.0
    oos_return_pct: float = 0.0
    oos_profit_factor: float = 0.0
    oos_score: float = 0.0


@dataclass
class OptimSummary:
    strategy: str = ""
    total_combos: int = 0
    tested: int = 0
    elapsed_sec: float = 0.0
    walk_forward: bool = False
    best: OptimResult = field(default_factory=OptimResult)
    top_n: list = field(default_factory=list)
    all_results: list = field(default_factory=list)
    param_keys: list = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_combos": self.total_combos,
            "tested": self.tested,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "walk_forward": self.walk_forward,
            "best": asdict(self.best),
            "top_n": [asdict(r) for r in self.top_n],
            "all_results": [
                {
                    "params": r.params,
                    "score": round(r.score, 4),
                    "pf": round(r.profit_factor, 2),
                    "sharpe": round(r.sharpe_ratio, 2),
                    "wr": round(r.win_rate, 4),
                    "ret": round(r.total_return_pct, 4),
                    "mdd": round(r.max_drawdown_pct, 4),
                    "trades": r.total_trades,
                    "rr": round(r.avg_rr_achieved, 2),
                    "oos_ret": round(r.oos_return_pct, 4),
                    "oos_pf": round(r.oos_profit_factor, 2),
                    "oos_score": round(r.oos_score, 4),
                }
                for r in self.all_results
            ],
            "param_keys": self.param_keys,
        }

    def export_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        logger.info(f"📄 최적화 결과 저장: {path}")


def compute_score(r: FastResult) -> float:
    if r.total_trades < 5:
        return 0.0
    pf = min(max(r.profit_factor, 0), 10)
    if pf > 50:
        return 0.0
    sh = max(r.sharpe_ratio, 0)
    mdd_pen = 1 - min(abs(r.max_drawdown_pct), 0.8)
    wr_bonus = min(r.win_rate / 0.3, 1.5)
    return round(pf * (sh ** 0.5) * mdd_pen * wr_bonus, 4)


def _run_single(args):
    strat, ohlcv, params, leverage, balance, sl_atr, rr, risk_pct, threshold, wf_ratio = args
    opens, highs, lows, closes, volumes = ohlcv
    n = len(closes)

    if wf_ratio and 0 < wf_ratio < 1:
        sp = int(n * wf_ratio)
        s_train, s_test = slice(0, sp), slice(sp, n)
    else:
        s_train, s_test = slice(0, n), None

    common = dict(leverage=leverage, risk_pct=risk_pct, sl_atr_mult=sl_atr,
                  rr_ratio=rr, initial_balance=balance)

    if strat in FAST_STRATEGIES:
        fn = FAST_STRATEGIES[strat]
        tr = fn(closes[s_train], highs[s_train], lows[s_train], volumes[s_train], params, **common)
    elif strat == "risk" or strat == "combined":
        trend_p = params.get("_trend", {"ema_fast": 9, "ema_slow": 21, "rsi_period": 14,
                                         "rsi_overbought": 70, "rsi_oversold": 30,
                                         "macd_fast": 12, "macd_slow": 26, "macd_signal": 9})
        break_p = params.get("_breakout", {"bb_period": 20, "bb_std": 2.0,
                                            "fib_lookback": 100, "volume_confirm_multiplier": 1.5})
        tr = fast_combined(closes[s_train], highs[s_train], lows[s_train], volumes[s_train],
                           trend_p, break_p, signal_threshold=threshold, **common)
    else:
        return OptimResult(params=params)

    sc = compute_score(tr)

    oos_ret = oos_pf = oos_sc = 0.0
    if s_test:
        if strat in FAST_STRATEGIES:
            te = FAST_STRATEGIES[strat](closes[s_test], highs[s_test], lows[s_test],
                                        volumes[s_test], params, **common)
        else:
            te = fast_combined(closes[s_test], highs[s_test], lows[s_test], volumes[s_test],
                               trend_p, break_p, signal_threshold=threshold, **common)
        oos_ret = te.total_return_pct
        oos_pf = te.profit_factor
        oos_sc = compute_score(te)

    return OptimResult(
        params={k: v for k, v in params.items() if not k.startswith("_")},
        total_trades=tr.total_trades, win_rate=tr.win_rate,
        total_return_pct=tr.total_return_pct, max_drawdown_pct=tr.max_drawdown_pct,
        profit_factor=tr.profit_factor, sharpe_ratio=tr.sharpe_ratio,
        avg_rr_achieved=tr.avg_rr_achieved, score=sc,
        oos_return_pct=oos_ret, oos_profit_factor=oos_pf, oos_score=oos_sc,
    )


def build_grid(pg, max_combos=5000):
    keys = sorted(pg.keys())
    vals = [pg[k] for k in keys]
    total = 1
    for v in vals:
        total *= len(v)
    if total <= max_combos:
        return [dict(zip(keys, c)) for c in product(*vals)]
    rng = np.random.default_rng(42)
    seen, out = set(), []
    while len(out) < max_combos:
        c = tuple(rng.choice(v).item() if hasattr(rng.choice(v), 'item') else rng.choice(v) for v in vals)
        if c not in seen:
            seen.add(c)
            out.append(dict(zip(keys, c)))
    return out


def optimize(strategy, ohlcv, param_grid, leverage=10, balance=1000.0,
             sl_atr=1.5, rr=3.0, risk_pct=0.02, threshold=0.6,
             walk_forward=None, max_workers=1, max_combos=5000, top_n=20):
    combos = build_grid(param_grid, max_combos)
    total = len(combos)
    logger.info(f"🔍 {strategy} | {total} combos | workers={max_workers}")

    tasks = [
        (strategy, ohlcv, c, leverage, balance, sl_atr, rr, risk_pct, threshold, walk_forward)
        for c in combos
    ]

    results = []
    t0 = time.time()

    if max_workers <= 1:
        for i, t in enumerate(tasks):
            results.append(_run_single(t))
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{total}")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for r in pool.map(_run_single, tasks, chunksize=max(1, total // (max_workers * 4))):
                results.append(r)

    elapsed = time.time() - t0
    sk = "oos_score" if walk_forward else "score"
    results.sort(key=lambda r: getattr(r, sk), reverse=True)

    best = results[0] if results else OptimResult()
    logger.info(f"✅ 완료 {elapsed:.1f}s | best score={best.score:.4f} PF={best.profit_factor:.2f} "
                f"WR={best.win_rate:.1%} MDD={best.max_drawdown_pct:.2%}")

    return OptimSummary(
        strategy=strategy, total_combos=total, tested=len(results),
        elapsed_sec=elapsed, walk_forward=bool(walk_forward),
        best=best, top_n=results[:top_n], all_results=results,
        param_keys=sorted(param_grid.keys()),
    )


def _gen_ohlcv(n=2000):
    np.random.seed(42)
    p = 68000.0
    o, h, l, c, v = [], [], [], [], []
    for _ in range(n):
        d = np.random.randn() * 150
        oo = p; cc = p + d
        hh = max(oo, cc) + abs(np.random.randn() * 80)
        ll = min(oo, cc) - abs(np.random.randn() * 80)
        o.append(oo); h.append(hh); l.append(ll); c.append(cc); v.append(abs(np.random.randn() * 500 + 1000))
        p = cc
    return tuple(np.array(x) for x in (o, h, l, c, v))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--strategy", required=True, choices=["trend_follow", "breakout", "risk"])
    pa.add_argument("--days", type=int, default=30)
    pa.add_argument("--timeframe", default="15m")
    pa.add_argument("--leverage", type=int, default=10)
    pa.add_argument("--balance", type=float, default=1000.0)
    pa.add_argument("--workers", type=int, default=1)
    pa.add_argument("--top", type=int, default=20)
    pa.add_argument("--max-combos", type=int, default=5000)
    pa.add_argument("--walk-forward", type=float, default=None)
    pa.add_argument("--quick", action="store_true")
    pa.add_argument("--demo", action="store_true")
    pa.add_argument("--export", default=None)
    a = pa.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    if a.demo:
        ohlcv = _gen_ohlcv(2000)
    else:
        from config import load_config
        from core.client import BitgetClient
        cfg = load_config()
        client = BitgetClient(cfg)
        tf_m = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        limit = min(a.days * (1440 // tf_m.get(a.timeframe, 15)), 1000)
        raw = client.fetch_ohlcv(timeframe=a.timeframe, limit=limit)
        d = np.array(raw, dtype=float)
        ohlcv = (d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5])

    if a.quick:
        grid = QUICK_GRIDS.get(a.strategy, STRATEGY_GRIDS.get(a.strategy, {}))
    elif a.strategy == "risk":
        grid = RISK_GRID
    else:
        grid = STRATEGY_GRIDS.get(a.strategy, {})

    summary = optimize(
        a.strategy, ohlcv, grid,
        leverage=a.leverage, balance=a.balance,
        walk_forward=a.walk_forward,
        max_workers=a.workers,
        max_combos=a.max_combos,
        top_n=a.top,
    )

    print("\n" + "=" * 72)
    print(f"🏆 Top 10 — {a.strategy}")
    print("=" * 72)
    for i, r in enumerate(summary.top_n[:10]):
        oos = f" OOS={r.oos_return_pct:+.2%}" if a.walk_forward else ""
        print(f"  #{i+1:2d} score={r.score:.3f} PF={r.profit_factor:.2f} "
              f"Sharpe={r.sharpe_ratio:.2f} WR={r.win_rate:.1%} "
              f"MDD={r.max_drawdown_pct:.2%} Ret={r.total_return_pct:+.2%}{oos}")
        print(f"       {r.params}")

    out = a.export or f"data/optim_{a.strategy}.json"
    summary.export_json(out)


if __name__ == "__main__":
    main()
