"""
벡터화 고속 백테스터 - 최적화 전용

기존 BacktestEngine은 매 바마다 indicators를 재계산 (O(N²)).
FastBacktest는 indicator를 1회 사전 계산 후 시그널만 루프 (O(N)).

500바 기준: 기존 ~3초 → Fast ~0.01초 (300x 속도 향상)
"""
import numpy as np
from dataclasses import dataclass, field
from strategy.base import ema, sma, rsi, macd, bollinger_bands, atr, fibonacci_levels


@dataclass
class FastResult:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_rr_achieved: float = 0.0


def fast_trend_follow(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    params: dict,
    leverage: int = 10,
    risk_pct: float = 0.02,
    sl_atr_mult: float = 1.5,
    rr_ratio: float = 3.0,
    initial_balance: float = 1000.0,
) -> FastResult:
    """
    트렌드 팔로잉 전략 벡터화 백테스트

    모든 인디케이터를 1회 계산 후 시그널 루프만 수행.
    """
    n = len(closes)
    if n < 50:
        return FastResult()

    # ── 인디케이터 사전 계산 (1회) ──
    ef = params.get("ema_fast", 9)
    es = params.get("ema_slow", 21)
    rp = params.get("rsi_period", 14)
    rob = params.get("rsi_overbought", 70)
    ros = params.get("rsi_oversold", 30)
    mf = params.get("macd_fast", 12)
    ms = params.get("macd_slow", 26)
    msig = params.get("macd_signal", 9)

    ema_f = ema(closes, ef)
    ema_s = ema(closes, es)
    rsi_v = rsi(closes, rp)
    _, _, hist = macd(closes, mf, ms, msig)
    atr_v = atr(highs, lows, closes, 14)

    # ── 시그널 벡터 생성 ──
    # EMA 크로스
    cross_up = (ema_f[1:] > ema_s[1:]) & (ema_f[:-1] <= ema_s[:-1])
    cross_down = (ema_f[1:] < ema_s[1:]) & (ema_f[:-1] >= ema_s[:-1])
    ema_above = ema_f > ema_s
    ema_below = ema_f < ema_s

    # MACD 전환
    macd_bull = np.zeros(n, dtype=bool)
    macd_bear = np.zeros(n, dtype=bool)
    macd_bull[1:] = (hist[1:] > 0) & (hist[:-1] <= 0)
    macd_bear[1:] = (hist[1:] < 0) & (hist[:-1] >= 0)

    # ── 트레이딩 시뮬레이션 ──
    balance = initial_balance
    equity = [balance]
    pnls = []
    in_pos = False
    pos_side = ""
    pos_entry = pos_sl = pos_tp = pos_amt = 0.0

    lb = max(30, es + 2)  # lookback

    for i in range(lb, n):
        c_high, c_low, c_close = highs[i], lows[i], closes[i]
        curr_atr = atr_v[i]
        curr_rsi = rsi_v[i]

        if np.isnan(curr_atr) or np.isnan(curr_rsi):
            equity.append(balance)
            continue

        # 포지션 보유 중 → SL/TP 체크
        if in_pos:
            if pos_side == "long":
                hit_sl = c_low <= pos_sl
                hit_tp = c_high >= pos_tp
            else:
                hit_sl = c_high >= pos_sl
                hit_tp = c_low <= pos_tp

            if hit_sl or hit_tp:
                ep = pos_sl if hit_sl else pos_tp
                if pos_side == "long":
                    raw_pnl = (ep - pos_entry) / pos_entry
                else:
                    raw_pnl = (pos_entry - ep) / pos_entry
                lev_pnl = raw_pnl * leverage * pos_amt
                balance *= (1 + lev_pnl)
                pnls.append(lev_pnl)
                in_pos = False

                if balance <= initial_balance * 0.05:
                    equity.append(balance)
                    break

            equity.append(balance)
            continue

        # 시그널 평가
        signal = 0  # +1 long, -1 short, 0 neutral

        # LONG 시그널
        if i > 0 and i - 1 < len(cross_up):
            is_cross_up = cross_up[i - 1]
        else:
            is_cross_up = False

        if is_cross_up and curr_rsi > 50 and curr_rsi < rob:
            signal = 1
        elif ema_above[i] and macd_bull[i] and 40 < curr_rsi < rob:
            signal = 1

        # SHORT 시그널
        if i > 0 and i - 1 < len(cross_down):
            is_cross_down = cross_down[i - 1]
        else:
            is_cross_down = False

        if is_cross_down and curr_rsi < 50 and curr_rsi > ros:
            signal = -1
        elif ema_below[i] and macd_bear[i] and ros < curr_rsi < 60:
            signal = -1

        if signal == 0:
            equity.append(balance)
            continue

        # 포지션 오픈
        sl_dist = curr_atr * sl_atr_mult
        if sl_dist <= 0:
            equity.append(balance)
            continue

        if signal == 1:
            pos_side = "long"
            pos_entry = c_close
            pos_sl = c_close - sl_dist
            pos_tp = c_close + sl_dist * rr_ratio
        else:
            pos_side = "short"
            pos_entry = c_close
            pos_sl = c_close + sl_dist
            pos_tp = c_close - sl_dist * rr_ratio

        sl_pct = sl_dist / c_close
        pos_amt = min(risk_pct / (sl_pct * leverage), 0.3) if sl_pct > 0 else 0.01
        in_pos = True
        equity.append(balance)

    return _compile_fast(pnls, equity, initial_balance, balance)


def fast_breakout(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    params: dict,
    leverage: int = 10,
    risk_pct: float = 0.02,
    sl_atr_mult: float = 1.5,
    rr_ratio: float = 3.0,
    initial_balance: float = 1000.0,
) -> FastResult:
    """볼린저 밴드 브레이크아웃 벡터화 백테스트"""
    n = len(closes)
    if n < 50:
        return FastResult()

    bp = params.get("bb_period", 20)
    bs = params.get("bb_std", 2.0)
    fl = params.get("fib_lookback", 100)
    vm = params.get("volume_confirm_multiplier", 1.5)

    upper, middle, lower = bollinger_bands(closes, bp, bs)
    atr_v = atr(highs, lows, closes, 14)
    vol_sma = sma(volumes, 20)

    balance = initial_balance
    equity = [balance]
    pnls = []
    in_pos = False
    pos_side = ""
    pos_entry = pos_sl = pos_tp = pos_amt = 0.0

    lb = max(30, bp + 2)

    for i in range(lb, n):
        c_high, c_low, c_close = highs[i], lows[i], closes[i]
        curr_atr = atr_v[i]

        if np.isnan(curr_atr) or np.isnan(upper[i]):
            equity.append(balance)
            continue

        if in_pos:
            if pos_side == "long":
                hit_sl = c_low <= pos_sl
                hit_tp = c_high >= pos_tp
            else:
                hit_sl = c_high >= pos_sl
                hit_tp = c_low <= pos_tp

            if hit_sl or hit_tp:
                ep = pos_sl if hit_sl else pos_tp
                raw_pnl = (ep - pos_entry) / pos_entry if pos_side == "long" \
                    else (pos_entry - ep) / pos_entry
                lev_pnl = raw_pnl * leverage * pos_amt
                balance *= (1 + lev_pnl)
                pnls.append(lev_pnl)
                in_pos = False
                if balance <= initial_balance * 0.05:
                    equity.append(balance)
                    break

            equity.append(balance)
            continue

        # 시그널
        signal = 0
        vol_ratio = volumes[i] / vol_sma[i] if vol_sma[i] > 0 and not np.isnan(vol_sma[i]) else 0
        bb_up_break = closes[i] > upper[i] and closes[i-1] <= upper[i-1]
        bb_dn_break = closes[i] < lower[i] and closes[i-1] >= lower[i-1]

        if bb_up_break and vol_ratio >= vm:
            signal = 1
        elif bb_dn_break and vol_ratio >= vm:
            signal = -1

        if signal == 0:
            equity.append(balance)
            continue

        sl_dist = curr_atr * sl_atr_mult
        if sl_dist <= 0:
            equity.append(balance)
            continue

        if signal == 1:
            pos_side, pos_entry = "long", c_close
            pos_sl = c_close - sl_dist
            pos_tp = c_close + sl_dist * rr_ratio
        else:
            pos_side, pos_entry = "short", c_close
            pos_sl = c_close + sl_dist
            pos_tp = c_close - sl_dist * rr_ratio

        sl_pct = sl_dist / c_close
        pos_amt = min(risk_pct / (sl_pct * leverage), 0.3) if sl_pct > 0 else 0.01
        in_pos = True
        equity.append(balance)

    return _compile_fast(pnls, equity, initial_balance, balance)


def fast_combined(
    closes, highs, lows, volumes,
    trend_params: dict, breakout_params: dict,
    trend_weight: float = 0.5, breakout_weight: float = 0.5,
    signal_threshold: float = 0.6,
    leverage: int = 10,
    risk_pct: float = 0.02,
    sl_atr_mult: float = 1.5,
    rr_ratio: float = 3.0,
    initial_balance: float = 1000.0,
) -> FastResult:
    """다중 전략 결합 벡터화 백테스트"""
    n = len(closes)
    if n < 50:
        return FastResult()

    # 트렌드 인디케이터
    ef = trend_params.get("ema_fast", 9)
    es = trend_params.get("ema_slow", 21)
    rp = trend_params.get("rsi_period", 14)
    rob = trend_params.get("rsi_overbought", 70)
    ros = trend_params.get("rsi_oversold", 30)

    ema_f = ema(closes, ef)
    ema_s = ema(closes, es)
    rsi_v = rsi(closes, rp)
    _, _, hist = macd(closes,
                      trend_params.get("macd_fast", 12),
                      trend_params.get("macd_slow", 26),
                      trend_params.get("macd_signal", 9))

    # 브레이크아웃 인디케이터
    bp = breakout_params.get("bb_period", 20)
    bs = breakout_params.get("bb_std", 2.0)
    vm = breakout_params.get("volume_confirm_multiplier", 1.5)
    upper, middle, lower = bollinger_bands(closes, bp, bs)
    vol_sma_v = sma(volumes, 20)
    atr_v = atr(highs, lows, closes, 14)

    balance = initial_balance
    equity = [balance]
    pnls = []
    in_pos = False
    pos_side = ""
    pos_entry = pos_sl = pos_tp = pos_amt = 0.0
    lb = max(35, es + 5)

    for i in range(lb, n):
        ch, cl, cc = highs[i], lows[i], closes[i]
        ca = atr_v[i]
        cr = rsi_v[i]

        if np.isnan(ca) or np.isnan(cr):
            equity.append(balance)
            continue

        if in_pos:
            hsl = (cl <= pos_sl) if pos_side == "long" else (ch >= pos_sl)
            htp = (ch >= pos_tp) if pos_side == "long" else (cl <= pos_tp)
            if hsl or htp:
                ep = pos_sl if hsl else pos_tp
                rp_ = (ep - pos_entry) / pos_entry if pos_side == "long" else (pos_entry - ep) / pos_entry
                lp = rp_ * leverage * pos_amt
                balance *= (1 + lp)
                pnls.append(lp)
                in_pos = False
                if balance <= initial_balance * 0.05:
                    equity.append(balance)
                    break
            equity.append(balance)
            continue

        # Trend score
        ts = 0.0
        cross_up = ema_f[i] > ema_s[i] and ema_f[i-1] <= ema_s[i-1]
        cross_dn = ema_f[i] < ema_s[i] and ema_f[i-1] >= ema_s[i-1]
        mb = hist[i] > 0 and hist[i-1] <= 0
        ms_ = hist[i] < 0 and hist[i-1] >= 0

        if cross_up and 50 < cr < rob:
            ts = 0.9 if mb else 0.7
        elif ema_f[i] > ema_s[i] and mb and 40 < cr < rob:
            ts = 0.65
        elif cross_dn and ros < cr < 50:
            ts = -0.9 if ms_ else -0.7
        elif ema_f[i] < ema_s[i] and ms_ and ros < cr < 60:
            ts = -0.65

        # Breakout score
        bs_ = 0.0
        vr = volumes[i] / vol_sma_v[i] if not np.isnan(vol_sma_v[i]) and vol_sma_v[i] > 0 else 0
        if not np.isnan(upper[i]):
            if cc > upper[i] and closes[i-1] <= upper[i-1] and vr >= vm:
                bs_ = 0.7
            elif cc < lower[i] and closes[i-1] >= lower[i-1] and vr >= vm:
                bs_ = -0.7

        # 결합
        combined = ts * trend_weight + bs_ * breakout_weight
        tw = trend_weight + breakout_weight
        if tw > 0:
            combined /= tw

        if abs(combined) < signal_threshold:
            equity.append(balance)
            continue

        sld = ca * sl_atr_mult
        if sld <= 0:
            equity.append(balance)
            continue

        if combined > 0:
            pos_side, pos_entry = "long", cc
            pos_sl = cc - sld
            pos_tp = cc + sld * rr_ratio
        else:
            pos_side, pos_entry = "short", cc
            pos_sl = cc + sld
            pos_tp = cc - sld * rr_ratio

        sp = sld / cc
        pos_amt = min(risk_pct / (sp * leverage), 0.3) if sp > 0 else 0.01
        in_pos = True
        equity.append(balance)

    return _compile_fast(pnls, equity, initial_balance, balance)


def _compile_fast(pnls, equity, initial_balance, final_balance) -> FastResult:
    """고속 결과 집계"""
    if not pnls:
        return FastResult()

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    aw = float(np.mean(wins)) if wins else 0
    al = float(np.mean(losses)) if losses else 0
    gp = sum(wins)
    gl = abs(sum(losses))

    eq = np.array(equity)
    pk = np.maximum.accumulate(eq)
    dd = (eq - pk) / pk

    rt = np.diff(eq) / eq[:-1]
    sh = float(np.mean(rt) / np.std(rt) * np.sqrt(365)) if len(rt) > 1 and np.std(rt) > 0 else 0.0

    return FastResult(
        total_trades=len(pnls),
        wins=len(wins), losses=len(losses),
        win_rate=len(wins) / len(pnls) if pnls else 0,
        total_return_pct=(final_balance - initial_balance) / initial_balance,
        max_drawdown_pct=float(np.min(dd)) if len(dd) > 0 else 0,
        profit_factor=gp / gl if gl > 0 else float("inf"),
        sharpe_ratio=sh,
        avg_win_pct=aw, avg_loss_pct=al,
        avg_rr_achieved=abs(aw / al) if al else 0,
    )


# 전략 이름 → fast 함수 매핑
FAST_STRATEGIES = {
    "trend_follow": fast_trend_follow,
    "breakout": fast_breakout,
}
