"""
OKX 실데이터 기반 전체 파라미터 최적화 + Walk-Forward 검증

실행:
  python backtest/run_full_search.py

출력:
  - 콘솔: Top 5 파라미터 조합 (전략별)
  - 파일: data/full_search_result.json
"""
import sys
import io
import time
import json
import logging
import numpy as np
from pathlib import Path

# Windows cp949 인코딩 에러 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from core.client import OKXClient
from backtest.optimizer import optimize, compute_score
from backtest.fast_backtest import fast_combined

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── 확장 파라미터 그리드 ──────────────────────────────────
TREND_GRID_FULL = {
    "ema_fast":       [3, 5, 7, 9, 12, 15],
    "ema_slow":       [15, 21, 26, 34, 50],
    "rsi_period":     [10, 14, 21],
    "rsi_overbought": [65, 70, 75, 80],
    "rsi_oversold":   [20, 25, 30, 35],
    "macd_fast":      [6, 8, 12],
    "macd_slow":      [21, 26],
    "macd_signal":    [7, 9, 12],
}

BREAKOUT_GRID_FULL = {
    "bb_period":                 [10, 14, 20, 25, 30],
    "bb_std":                    [1.0, 1.5, 2.0, 2.5, 3.0],
    "fib_lookback":              [50, 100, 150, 200],
    "volume_confirm_multiplier": [1.0, 1.2, 1.5, 2.0, 2.5],
}

RISK_SL_MULT   = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
RISK_RR        = [2.0, 2.5, 3.0, 4.0, 5.0]
RISK_THRESHOLD = [0.25, 0.30, 0.35, 0.42, 0.50]

LEVERAGE_TEST = 8


def fetch_ohlcv_extended(client: OKXClient, timeframe: str = "15m", days: int = 90) -> tuple:
    """CCXT since 기반 순방향 페이지네이션으로 장기 OHLCV 수집"""
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
    mins = tf_minutes.get(timeframe, 15)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000

    logger.info(f"OKX {timeframe} 캔들 수집 시작 (목표 {days}일치)...")

    all_data = []
    batch = 300
    current_since = start_ms
    candle_ms = mins * 60 * 1000
    attempt = 0

    while current_since < now_ms:
        attempt += 1
        try:
            raw = client.exchange.fetch_ohlcv(
                client.config.trading["symbol"],
                timeframe=timeframe,
                limit=batch,
                since=current_since,
            )

            if not raw:
                logger.info(f"  배치 {attempt}: 빈 응답, 수집 종료")
                break

            all_data.extend(raw)
            last_ts = raw[-1][0]
            logger.info(
                f"  배치 {attempt}: {len(raw)}개 | 누적 {len(all_data)}개"
            )

            current_since = last_ts + candle_ms

            if len(raw) < batch:
                logger.info("  마지막 배치, 수집 완료")
                break

            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"  배치 {attempt} 실패: {e}")
            break

    # 중복 제거 + 시간순 정렬
    seen = set()
    unique = []
    for bar in all_data:
        if bar[0] not in seen:
            seen.add(bar[0])
            unique.append(bar)
    unique.sort(key=lambda x: x[0])

    if len(unique) < 100:
        logger.error(f"데이터 부족 ({len(unique)}개)")
        return None

    d = np.array(unique, dtype=float)
    logger.info(f"최종 {len(d)}개 캔들 (~{len(d) * mins // 1440}일)")
    return (d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5])


def run_search(ohlcv: tuple) -> dict:
    results = {}

    # ── 1. Trend Follow ────────────────────────────────────
    print("\n" + "=" * 60)
    print("[1/3] Trend Follow 최적화 (Walk-Forward 70/30)")
    print("=" * 60)
    tf_summary = optimize(
        strategy="trend_follow",
        ohlcv=ohlcv,
        param_grid=TREND_GRID_FULL,
        leverage=LEVERAGE_TEST,
        balance=1000.0,
        sl_atr=1.5,
        rr=3.0,
        risk_pct=0.02,
        walk_forward=0.7,
        max_workers=1,
        max_combos=5000,
        top_n=10,
    )
    results["trend_follow"] = tf_summary.to_json()
    _print_top5(tf_summary, "Trend Follow")

    # ── 2. Breakout ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2/3] Breakout 최적화 (Walk-Forward 70/30)")
    print("=" * 60)
    bo_summary = optimize(
        strategy="breakout",
        ohlcv=ohlcv,
        param_grid=BREAKOUT_GRID_FULL,
        leverage=LEVERAGE_TEST,
        balance=1000.0,
        sl_atr=1.5,
        rr=3.0,
        risk_pct=0.02,
        walk_forward=0.7,
        max_workers=1,
        max_combos=1000,
        top_n=10,
    )
    results["breakout"] = bo_summary.to_json()
    _print_top5(bo_summary, "Breakout")

    # ── 3. SL/TP/Threshold ─────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/3] SL / TP / Signal Threshold 최적화")
    print("=" * 60)

    best_trend = tf_summary.best.params
    best_break = bo_summary.best.params

    n = len(ohlcv[2])
    sp = int(n * 0.7)
    train = tuple(x[:sp] for x in ohlcv)
    test  = tuple(x[sp:] for x in ohlcv)

    risk_results = []
    total = len(RISK_SL_MULT) * len(RISK_RR) * len(RISK_THRESHOLD)
    done = 0
    for sl_m in RISK_SL_MULT:
        for rr in RISK_RR:
            for thr in RISK_THRESHOLD:
                done += 1
                if done % 10 == 0:
                    logger.info(f"  {done}/{total}")

                kw = dict(
                    trend_params=best_trend,
                    breakout_params=best_break,
                    trend_weight=0.7, breakout_weight=0.3,
                    signal_threshold=thr,
                    leverage=LEVERAGE_TEST,
                    sl_atr_mult=sl_m,
                    rr_ratio=rr,
                    risk_pct=0.02,
                )
                tr = fast_combined(train[1], train[2], train[3], train[4], **kw)
                te = fast_combined(test[1],  test[2],  test[3],  test[4],  **kw)
                sc     = compute_score(tr)
                oos_sc = compute_score(te)
                risk_results.append({
                    "sl_atr_mult": sl_m,
                    "rr_ratio": rr,
                    "signal_threshold": thr,
                    "train_return": round(tr.total_return_pct, 4),
                    "train_pf":     round(tr.profit_factor, 2),
                    "train_wr":     round(tr.win_rate, 4),
                    "train_mdd":    round(tr.max_drawdown_pct, 4),
                    "train_trades": tr.total_trades,
                    "oos_return":   round(te.total_return_pct, 4),
                    "oos_pf":       round(te.profit_factor, 2),
                    "oos_wr":       round(te.win_rate, 4),
                    "oos_mdd":      round(te.max_drawdown_pct, 4),
                    "train_score":  round(sc, 4),
                    "oos_score":    round(oos_sc, 4),
                })

    risk_results.sort(key=lambda x: x["oos_score"], reverse=True)
    results["risk_params"] = risk_results
    _print_risk_top5(risk_results)

    return results


def _print_top5(summary, label: str):
    print(f"\n--- TOP5: {label} (OOS 기준) ---")
    for i, r in enumerate(summary.top_n[:5]):
        print(
            f"  #{i+1}  OOS수익={r.oos_return_pct:+.2%}  OOS PF={r.oos_profit_factor:.2f}  "
            f"IS WR={r.win_rate:.1%}  IS MDD={r.max_drawdown_pct:.2%}  거래={r.total_trades}회"
        )
        print(f"       {r.params}")
    print()


def _print_risk_top5(results: list):
    print(f"\n--- TOP5: SL/TP/Threshold (OOS 기준) ---")
    for i, r in enumerate(results[:5]):
        print(
            f"  #{i+1}  OOS수익={r['oos_return']:+.2%}  OOS PF={r['oos_pf']:.2f}  "
            f"OOS WR={r['oos_wr']:.1%}  OOS MDD={r['oos_mdd']:.2%}"
        )
        print(
            f"       SL={r['sl_atr_mult']}xATR  TP=1:{r['rr_ratio']}  "
            f"threshold={r['signal_threshold']}"
        )
    print()


def main():
    logger.info("=" * 60)
    logger.info("OKX 실데이터 전체 파라미터 최적화 시작")
    logger.info("=" * 60)

    cfg = load_config()
    client = OKXClient(cfg)

    ohlcv = fetch_ohlcv_extended(client, timeframe="15m", days=90)
    if ohlcv is None:
        logger.error("데이터 수집 실패")
        return

    n = len(ohlcv[2])
    logger.info(f"백테스트 데이터: {n}개 캔들 (~{n * 15 // 1440}일)")

    t0 = time.time()
    results = run_search(ohlcv)
    elapsed = time.time() - t0
    logger.info(f"총 소요시간: {elapsed:.1f}초")

    out_path = Path("data/full_search_result.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"결과 저장: {out_path}")

    # 최종 추천
    print("\n" + "=" * 60)
    print("  [최종 추천 설정 — settings.yaml 적용 가이드]")
    print("=" * 60)

    tf_best = results["trend_follow"]["best"]
    bo_best = results["breakout"]["best"]
    rk_best = results["risk_params"][0] if results.get("risk_params") else {}

    print("\n  [trend_follow]")
    for k, v in tf_best.get("params", {}).items():
        print(f"    {k}: {v}")

    print("\n  [breakout]")
    for k, v in bo_best.get("params", {}).items():
        print(f"    {k}: {v}")

    if rk_best:
        print("\n  [risk / signal]")
        print(f"    stop_loss_atr_multiplier: {rk_best.get('sl_atr_mult')}")
        print(f"    take_profit_rr_ratio:     {rk_best.get('rr_ratio')}")
        print(f"    signal_threshold:         {rk_best.get('signal_threshold')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
