"""
전략별 파라미터 그리드 정의

각 전략의 탐색 범위를 정의하고, 리스크/글로벌 파라미터 그리드도 포함.
그리드 사이즈가 너무 크면 random sampling으로 전환됨.
"""

# ── Trend Follow ──────────────────────────────────────
TREND_FOLLOW_GRID = {
    "ema_fast":       [5, 7, 9, 12],
    "ema_slow":       [15, 21, 26, 34],
    "rsi_period":     [10, 14, 21],
    "rsi_overbought": [65, 70, 75],
    "rsi_oversold":   [25, 30, 35],
    "macd_fast":      [8, 12],
    "macd_slow":      [21, 26],
    "macd_signal":    [7, 9],
}
# 4 × 4 × 3 × 3 × 3 × 2 × 2 × 2 = 3,456 조합

# ── Breakout ──────────────────────────────────────────
BREAKOUT_GRID = {
    "bb_period":                   [14, 20, 25],
    "bb_std":                      [1.5, 2.0, 2.5, 3.0],
    "fib_lookback":                [50, 100, 150],
    "volume_confirm_multiplier":   [1.2, 1.5, 2.0],
}
# 3 × 4 × 3 × 3 = 108 조합

# ── Scalping ──────────────────────────────────────────
SCALPING_GRID = {
    "orderbook_depth":       [10, 15, 20, 30],
    "imbalance_threshold":   [0.55, 0.60, 0.65, 0.70],
    "min_spread_bps":        [2, 3, 5],
}
# 4 × 4 × 3 = 48 조합

# ── 글로벌 (리스크 + 전략 가중치) ─────────────────────
RISK_GRID = {
    "stop_loss_atr_multiplier": [1.0, 1.5, 2.0, 2.5],
    "take_profit_rr_ratio":     [2.0, 2.5, 3.0, 4.0, 5.0],
    "risk_per_trade_pct":       [0.01, 0.02, 0.03],
    "signal_threshold":         [0.4, 0.5, 0.6, 0.7],
}
# 4 × 5 × 3 × 4 = 240 조합

WEIGHT_GRID = {
    "trend_follow": [0.3, 0.4, 0.5],
    "breakout":     [0.2, 0.3, 0.4],
    "scalping":     [0.1, 0.2, 0.3],
}

# ── 전략 이름 → 그리드 매핑 ───────────────────────────
STRATEGY_GRIDS = {
    "trend_follow": TREND_FOLLOW_GRID,
    "breakout":     BREAKOUT_GRID,
    "scalping":     SCALPING_GRID,
}

# ── 프리셋: 빠른 탐색용 축소 그리드 ──────────────────
TREND_FOLLOW_QUICK = {
    "ema_fast":       [7, 9, 12],
    "ema_slow":       [21, 26],
    "rsi_period":     [14],
    "rsi_overbought": [70],
    "rsi_oversold":   [30],
    "macd_fast":      [12],
    "macd_slow":      [26],
    "macd_signal":    [9],
}

BREAKOUT_QUICK = {
    "bb_period":                   [20],
    "bb_std":                      [1.5, 2.0, 2.5],
    "fib_lookback":                [100],
    "volume_confirm_multiplier":   [1.5, 2.0],
}

RISK_QUICK = {
    "stop_loss_atr_multiplier": [1.0, 1.5, 2.0],
    "take_profit_rr_ratio":     [2.0, 3.0, 4.0],
    "risk_per_trade_pct":       [0.02],
    "signal_threshold":         [0.5, 0.6],
}

QUICK_GRIDS = {
    "trend_follow": TREND_FOLLOW_QUICK,
    "breakout":     BREAKOUT_QUICK,
    "risk":         RISK_QUICK,
}
