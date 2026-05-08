"""
시나리오 비교 백테스터 - 월 50% 목표 최적화
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
import numpy as np
from datetime import datetime, timezone
from strategy.base import ema, rsi, macd, atr


def calc_adx(highs, lows, closes, period=14):
    n = len(closes)
    tr = np.zeros(n)
    pdm = np.zeros(n)
    ndm = np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        up   = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        pdm[i] = up   if up > down and up > 0   else 0
        ndm[i] = down if down > up and down > 0 else 0

    def wilder(arr, p):
        out = np.zeros(n)
        if p < n:
            out[p] = arr[1:p+1].sum()
        for i in range(p+1, n):
            out[i] = out[i-1] - out[i-1]/p + arr[i]
        return out

    atr_w = wilder(tr, period)
    pdm_w = wilder(pdm, period)
    ndm_w = wilder(ndm, period)
    pdi = np.where(atr_w > 0, 100*pdm_w/atr_w, 0)
    ndi = np.where(atr_w > 0, 100*ndm_w/atr_w, 0)
    dx  = np.where((pdi+ndi) > 0, 100*np.abs(pdi-ndi)/(pdi+ndi), 0)
    adx_v = wilder(dx, period) / period
    return adx_v


def run_bt(
    closes, highs, lows, vols, ts,
    closes_1h, ts_1h,
    rr_strong=4.0,
    rr_normal=3.0,
    partial_tp=True,
    trail_callback=1.5,
    adx_filter=20,
    vol_min=0.5,
    sl_atr=1.5,
    leverage=8,
    risk_pct=0.02,
    initial_balance=160.0,
    label='',
):
    n = len(closes)

    ema_f  = ema(closes, 5)
    ema_s  = ema(closes, 26)
    rsi_v  = rsi(closes, 14)
    _, _, hist = macd(closes, 8, 26, 7)
    atr_v  = atr(highs, lows, closes, 14)
    adx_v  = calc_adx(highs, lows, closes, 14)

    vol_sma = np.array([
        float(np.mean(vols[max(0,i-20):i])) if i >= 20 else 0.0
        for i in range(n)
    ])

    ema_1h_f = ema(closes_1h, 9)
    ema_1h_s = ema(closes_1h, 26)
    ts_1h_arr = np.array(ts_1h)

    def trend_1h(ts_ms):
        idx = int(np.searchsorted(ts_1h_arr, ts_ms, side='right')) - 1
        if idx < 26:
            return 'unknown'
        if np.isnan(ema_1h_f[idx]) or np.isnan(ema_1h_s[idx]):
            return 'unknown'
        return 'above' if ema_1h_f[idx] > ema_1h_s[idx] else 'below'

    balance = initial_balance
    pnls = []
    monthly = {}
    equity = [balance]

    in_pos = False
    pos_side = ''
    pos_entry = pos_sl = pos_tp1 = pos_tp2 = 0.0
    pos_amt_full = 0.0
    pos_partial_done = False
    pos_hwm = pos_lwm = 0.0

    day_str = ''
    day_bal = initial_balance
    daily_n = 0
    consec_loss = 0
    cooldown_bar = 0

    for i in range(28, n):
        cc = closes[i]
        ch = highs[i]
        cl = lows[i]
        cur_atr  = atr_v[i]
        cur_rsi  = rsi_v[i]
        cur_hist = hist[i]
        prv_hist = hist[i-1]
        cur_adx  = adx_v[i]

        if np.isnan(cur_atr) or np.isnan(cur_rsi) or np.isnan(cur_hist):
            equity.append(balance)
            continue

        ts_ms   = ts[i]
        dt      = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
        day_s   = dt.strftime('%Y-%m-%d')
        month_s = dt.strftime('%Y-%m')

        if day_s != day_str:
            day_str = day_s
            day_bal = balance
            daily_n = 0

        # ── 포지션 모니터링 ──
        if in_pos:
            cb = trail_callback / 100

            # 부분 청산 (1.5R)
            if partial_tp and not pos_partial_done:
                hit1 = (pos_side == 'long' and ch >= pos_tp1) or \
                       (pos_side == 'short' and cl <= pos_tp1)
                if hit1:
                    ep = pos_tp1
                    raw = (ep-pos_entry)/pos_entry if pos_side == 'long' \
                          else (pos_entry-ep)/pos_entry
                    bal_chg = raw * pos_amt_full * 0.5
                    balance *= (1 + bal_chg)
                    pnls.append(bal_chg)
                    monthly[month_s] = monthly.get(month_s, 0) + bal_chg
                    pos_partial_done = True
                    pos_hwm = ch
                    pos_lwm = cl

            if pos_partial_done:
                pos_hwm = max(pos_hwm, ch)
                pos_lwm = min(pos_lwm, cl)

            closed = False
            exit_p = 0.0

            if pos_side == 'long':
                if cl <= pos_sl:
                    closed = True
                    exit_p = pos_sl
                elif partial_tp and pos_partial_done:
                    trail_sl = pos_hwm * (1 - cb)
                    if cl <= trail_sl:
                        closed = True
                        exit_p = trail_sl
                elif ch >= pos_tp2:
                    closed = True
                    exit_p = pos_tp2
            else:
                if ch >= pos_sl:
                    closed = True
                    exit_p = pos_sl
                elif partial_tp and pos_partial_done:
                    trail_sl = pos_lwm * (1 + cb)
                    if ch >= trail_sl:
                        closed = True
                        exit_p = trail_sl
                elif cl <= pos_tp2:
                    closed = True
                    exit_p = pos_tp2

            if closed:
                raw = (exit_p-pos_entry)/pos_entry if pos_side == 'long' \
                      else (pos_entry-exit_p)/pos_entry
                remain = 0.5 if pos_partial_done else 1.0
                bal_chg = raw * pos_amt_full * remain
                balance *= (1 + bal_chg)
                pnls.append(bal_chg)
                monthly[month_s] = monthly.get(month_s, 0) + bal_chg

                if bal_chg < 0:
                    consec_loss += 1
                    if consec_loss >= 3:
                        cooldown_bar = i + 4
                        consec_loss = 0
                else:
                    consec_loss = 0

                in_pos = False
                if balance <= initial_balance * 0.1:
                    equity.append(balance)
                    break

            equity.append(balance)
            continue

        # ── 가드 ──
        if i < cooldown_bar:
            equity.append(balance)
            continue
        if (balance - day_bal) / day_bal < -0.05:
            equity.append(balance)
            continue
        if daily_n >= 10:
            equity.append(balance)
            continue

        # ── 거래량 필터 ──
        if vol_sma[i] > 0 and vols[i] / vol_sma[i] < vol_min:
            equity.append(balance)
            continue

        # ── ADX 필터 ──
        if adx_filter > 0 and cur_adx < adx_filter:
            equity.append(balance)
            continue

        # ── 시그널 ──
        ef_i = ema_f[i]; es_i = ema_s[i]
        ef_p = ema_f[i-1]; es_p = ema_s[i-1]
        cross_up = ef_p <= es_p and ef_i > es_i
        cross_dn = ef_p >= es_p and ef_i < es_i
        above = ef_i > es_i
        below = ef_i < es_i
        bull_x = prv_hist < 0 and cur_hist > 0
        bear_x = prv_hist > 0 and cur_hist < 0
        m_pos  = cur_hist > 0
        m_neg  = cur_hist < 0
        m_imp  = cur_hist > prv_hist
        m_wrs  = cur_hist < prv_hist

        sv = 0.0
        conf = 0.0
        strong = False

        if cross_up and 50 < cur_rsi < 70:
            if bull_x:   sv, conf, strong = 1.0, 0.95, True
            elif m_pos:  sv, conf, strong = 1.0, 0.85, True
            else:        sv, conf = 0.6, 0.75
        elif above and 40 < cur_rsi < 70:
            if bull_x:   sv, conf = 0.6, 0.85
            elif m_pos:  sv, conf = 0.6, 0.70
            elif m_imp and cur_rsi > 55 and prv_hist < 0 and cur_hist > prv_hist * 0.7:
                sv, conf = 0.6, 0.86
        elif cross_dn and cur_rsi < 50:
            if bear_x:   sv, conf, strong = -1.0, 0.95, True
            elif m_neg:  sv, conf, strong = -1.0, 0.85, True
            else:        sv, conf = -0.6, 0.75
        elif below and cur_rsi < 60:
            if bear_x:   sv, conf = -0.6, 0.85
            elif m_neg:  sv, conf = -0.6, 0.70
            elif m_wrs and cur_rsi < 43 and prv_hist > 0 \
                    and cur_hist < prv_hist * 0.6 and cur_hist < 5:
                sv, conf = -0.6, 0.86

        score = sv * conf
        if abs(score) < 0.5:
            equity.append(balance)
            continue

        # ── 1h 추세 필터 ──
        t1h = trend_1h(ts_ms)
        if score > 0 and t1h == 'below':
            equity.append(balance)
            continue
        if score < 0 and t1h == 'above':
            equity.append(balance)
            continue

        # ── 포지션 오픈 ──
        sl_d = cur_atr * sl_atr
        if sl_d <= 0:
            equity.append(balance)
            continue

        sl_pct = sl_d / cc
        pos_val = min(balance * risk_pct / sl_pct, balance * 1.0 * leverage)
        pos_amt_full = pos_val / balance

        rr = rr_strong if strong else rr_normal

        if score > 0:
            pos_side  = 'long'
            pos_entry = cc
            pos_sl    = cc - sl_d
            pos_tp1   = cc + sl_d * 1.5
            pos_tp2   = cc + sl_d * rr
            pos_hwm   = ch
        else:
            pos_side  = 'short'
            pos_entry = cc
            pos_sl    = cc + sl_d
            pos_tp1   = cc - sl_d * 1.5
            pos_tp2   = cc - sl_d * rr
            pos_lwm   = cl

        pos_partial_done = False
        in_pos = True
        daily_n += 1
        equity.append(balance)

    if not pnls:
        return None

    months = n / (4 * 24 * 30)
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    eq     = np.array(equity)
    pk     = np.maximum.accumulate(eq)
    dd     = float(np.min((eq - pk) / pk))
    total  = (balance - initial_balance) / initial_balance
    bd     = sorted(
        [{'month': m, 'pnl': v} for m, v in monthly.items()],
        key=lambda x: x['month']
    )
    return dict(
        label=label,
        trades=len(pnls),
        tpm=len(pnls)/months,
        wr=len(wins)/len(pnls) if pnls else 0,
        pf=sum(wins)/abs(sum(losses)) if losses else 99,
        total=total,
        monthly=total/months,
        dd=dd,
        balance=balance,
        monthly_bd=bd,
    )


if __name__ == '__main__':
    closes    = np.load('/tmp/bt_closes.npy')
    highs     = np.load('/tmp/bt_highs.npy')
    lows      = np.load('/tmp/bt_lows.npy')
    vols      = np.load('/tmp/bt_vols.npy')
    closes_1h = np.load('/tmp/bt_closes_1h.npy')
    with open('/tmp/bt_ts.json')    as f: ts    = json.load(f)
    with open('/tmp/bt_ts_1h.json') as f: ts_1h = json.load(f)

    scenarios = [
        dict(label='[현재봇] 기준',          rr_strong=3.0, rr_normal=3.0, partial_tp=False, adx_filter=0,  vol_min=0.5, trail_callback=1.5),
        dict(label='[A] 부분청산(1.5R 50%)', rr_strong=3.0, rr_normal=3.0, partial_tp=True,  adx_filter=0,  vol_min=0.5, trail_callback=1.5),
        dict(label='[B] STRONG RR=4',        rr_strong=4.0, rr_normal=3.0, partial_tp=False, adx_filter=0,  vol_min=0.5, trail_callback=1.5),
        dict(label='[C] ADX>=20 필터',       rr_strong=3.0, rr_normal=3.0, partial_tp=False, adx_filter=20, vol_min=0.5, trail_callback=1.5),
        dict(label='[D] A+B+C 조합',         rr_strong=4.0, rr_normal=3.0, partial_tp=True,  adx_filter=20, vol_min=0.5, trail_callback=1.0),
        dict(label='[E] 공격조합(콜백0.8%)', rr_strong=5.0, rr_normal=4.0, partial_tp=True,  adx_filter=20, vol_min=0.3, trail_callback=0.8),
        dict(label='[F] SL타이트(ATR×1.0)', rr_strong=4.0, rr_normal=3.0, partial_tp=True,  adx_filter=20, vol_min=0.5, trail_callback=1.0, sl_atr=1.0),
    ]

    print()
    print(f"{'시나리오':<22} | {'월거래':>5} | {'승률':>5} | {'손익비':>5} | {'월수익':>7} | {'최대낙폭':>8} | {'최종잔고':>9}")
    print('-' * 85)

    best = None
    for s in scenarios:
        r = run_bt(closes, highs, lows, vols, ts, closes_1h, ts_1h, **s)
        if r:
            mark = ' ★' if r['monthly'] >= 0.50 else ''
            print(
                f"{r['label']:<22} | {r['tpm']:>5.1f}회 | {r['wr']:>5.1%} | "
                f"{r['pf']:>5.2f} | {r['monthly']:>+7.2%} | {r['dd']:>8.2%} | "
                f"{r['balance']:>8.0f}USDT{mark}"
            )
            if best is None or r['monthly'] > best['monthly']:
                best = r

    if best:
        print()
        print(f"최고 시나리오: {best['label']}  (월평균 {best['monthly']:+.2%})")
        print()
        print('월별 수익률:')
        for m in best['monthly_bd']:
            p = m['pnl']
            bar = ('+' if p > 0 else '-') * min(int(abs(p) * 150), 30)
            print(f"  {m['month']}: {p:+.2%}  {bar}")
