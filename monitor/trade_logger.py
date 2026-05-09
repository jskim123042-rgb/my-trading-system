"""
거래 데이터 수집기 - SQLite 기반
3개월 데이터 축적 후 전략 최적화에 활용

수집 항목:
- signals : 모든 신호 평가 + 시장 컨텍스트 (진입 안 된 것도 포함)
- trades  : 진입/청산 전체 + MFE/MAE + 상위 추세 + 거래량 + 펀딩비
- daily   : 일간 스냅샷
"""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("data/trades.db")


class TradeLogger:

    def __init__(self):
        DB_PATH.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self._create_tables()
        logger.info(f"📁 거래 로그 DB: {DB_PATH.resolve()}")

    def _create_tables(self):
        cur = self.conn.cursor()

        # ── signals: 신호 평가 로그 ──────────────────────
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            weekday         INTEGER,        -- 0=월 6=일
            hour_utc        INTEGER,
            price           REAL,
            -- 15분봉 지표
            ema_fast        REAL,
            ema_slow        REAL,
            ema_trend       TEXT,           -- 'above'|'below'
            macd_hist       REAL,
            macd_prev       REAL,
            macd_state      TEXT,
            rsi             REAL,
            atr             REAL,
            atr_pct         REAL,           -- ATR/price (변동성 %)
            -- 상위 타임프레임 추세 (1h)
            ema_1h_trend    TEXT,           -- 'above'|'below'|'unknown'
            rsi_1h          REAL,
            -- 거래량
            volume_ratio    REAL,           -- 현재 거래량 / 20봉 평균 (1.0=평균)
            -- 시장 심리
            funding_rate    REAL,
            -- 신호
            score           REAL,
            direction       TEXT,           -- 'long'|'short'|'neutral'
            fired           INTEGER,        -- 1=진입 0=미달
            reasons         TEXT,           -- JSON
            -- ── 3개월 분석용 다이버전스 상세 ────────────────
            div_grade       TEXT,           -- A/B/C/D/E/none
            div_tf_count    INTEGER,        -- 확인된 TF 수 (0~3)
            div_strength    INTEGER,        -- 다이버전스 강도 (0~4)
            cvd_confirmed   INTEGER,        -- CVD 방향 일치 (0/1)
            bb_squeeze      INTEGER,        -- 볼린저 스퀴즈 (0/1)
            rsi_4h          REAL,           -- 4h RSI
            score_trend     REAL,           -- trend_follow 개별 기여 점수
            score_breakout  REAL            -- breakout 개별 기여 점수
        )
        """)

        # ── trades: 완료된 트레이드 ───────────────────────
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              TEXT PRIMARY KEY,
            open_ts         TEXT,
            close_ts        TEXT,
            side            TEXT,
            entry_price     REAL,
            exit_price      REAL,
            amount          REAL,
            stop_loss       REAL,
            take_profit     REAL,
            sl_pct          REAL,
            leverage        INTEGER,
            pnl_usdt        REAL,
            pnl_pct         REAL,
            exit_reason     TEXT,
            duration_min    REAL,
            -- 진입 시점 지표 (15분봉)
            entry_ema_fast  REAL,
            entry_ema_slow  REAL,
            entry_macd_hist REAL,
            entry_macd_prev REAL,
            entry_rsi       REAL,
            entry_atr       REAL,
            entry_score     REAL,
            entry_reasons   TEXT,
            -- 상위 추세 (1h / 4h)
            entry_1h_trend  TEXT,           -- 'above'|'below'|'unknown'
            entry_4h_trend  TEXT,
            entry_rsi_1h    REAL,
            -- 거래량 / 시장 심리
            entry_vol_ratio REAL,           -- 거래량 비율
            entry_funding   REAL,           -- 펀딩비
            -- 포지션 중 최대 이동 (SL/TP 최적화 핵심)
            mfe_pct         REAL,           -- Max Favorable Excursion (최대 유리 이동 %)
            mae_pct         REAL,           -- Max Adverse Excursion (최대 불리 이동 %)
            mfe_r           REAL,           -- MFE in R multiples (손절폭 대비 최대 수익 배수)
            -- 시장 컨텍스트
            weekday         INTEGER,
            hour_utc        INTEGER,
            -- ── 3개월 분석용 다이버전스 상세 ────────────────
            div_grade       TEXT,           -- A/B/C/D/E/none
            div_tf_count    INTEGER,        -- 확인된 TF 수 (0~3)
            div_strength    INTEGER,        -- 다이버전스 강도 (0~4)
            cvd_confirmed   INTEGER,        -- CVD 방향 일치 (0/1)
            entry_bb_squeeze INTEGER,       -- 볼린저 스퀴즈 후 진입 (0/1)
            entry_rsi_4h    REAL,           -- 진입 시 4h RSI
            score_trend     REAL,           -- trend_follow 개별 기여 점수
            score_breakout  REAL,           -- breakout 개별 기여 점수
            candle_body_pct REAL,           -- 진입 캔들 몸통 비율 (0~1, 강도 지표)
            -- ── 최적화 로그 추가 ─────────────────────────────
            bb_pct_b        REAL,           -- 진입 시 BB 위치 (0=하단, 1=상단)
            div_bars_between INTEGER,       -- 다이버전스 피벗 간 봉 수
            early_direction INTEGER,        -- 진입 후 첫 3봉 방향 일치 (1=맞음, 0=역행, NULL=미확인)
            entry_adx       REAL            -- 진입 시 ADX (추세강도, <25=횡보, >25=추세)
        )
        """)

        # ── shadow_trades: 포지션 중 무시된 반대 시그널 가상 추적 ──
        cur.execute("""
        CREATE TABLE IF NOT EXISTS shadow_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,          -- 시그널 발생 시각
            direction       TEXT,                   -- 가상 진입 방향 (long/short)
            score           REAL,                   -- 시그널 점수
            signal_price    REAL,                   -- 시그널 발생 시 가격
            stop_loss       REAL,
            take_profit     REAL,
            sl_pct          REAL,
            reasons         TEXT,                   -- JSON
            -- 당시 보유 포지션 정보
            active_side     TEXT,                   -- 당시 실제 포지션 방향
            active_entry    REAL,                   -- 당시 실제 진입가
            active_pnl_pct  REAL,                   -- 당시 실제 포지션 미실현 손익%
            -- 가상 결과 (사후 업데이트)
            outcome         TEXT,                   -- 'tp'|'sl'|'open'
            exit_price      REAL,
            exit_ts         TEXT,
            virtual_pnl_pct REAL,                   -- 가상 손익% (레버리지 적용)
            duration_min    REAL
        )
        """)

        # ── volume_profile: 거래량 스냅샷 (5분마다) ─────────
        cur.execute("""
        CREATE TABLE IF NOT EXISTS volume_profile (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            price           REAL,
            vol_current     REAL,           -- 현재봉 거래량
            vol_sma20       REAL,           -- 20봉 평균 거래량
            vol_ratio       REAL,           -- vol_current / vol_sma20
            vol_spike       INTEGER,        -- 1=급등(>2x), 0=일반
            vol_dry         INTEGER,        -- 1=거래량 적음(<0.5x)
            vol_trend_5     REAL,           -- 직전 5봉 거래량 추세 %
            buy_vol_pct     REAL,           -- 15분 매수 비율 (0~1)
            vol_delta       REAL,           -- 현재 캔들 델타 (매수-매도)
            vol_delta_cum   REAL,           -- 누적 CVD (5시간)
            cvd_slope_5     REAL,           -- CVD 기울기 (5봉)
            oi              REAL,           -- 미결제약정
            oi_change_pct   REAL,           -- OI 변화율 %
            long_short_ratio REAL           -- 롱숏 비율
        )
        """)

        # ── daily: 일간 스냅샷 ────────────────────────────
        cur.execute("""
        CREATE TABLE IF NOT EXISTS daily (
            date            TEXT PRIMARY KEY,
            start_balance   REAL,
            end_balance     REAL,
            total_trades    INTEGER,
            wins            INTEGER,
            losses          INTEGER,
            pnl_usdt        REAL,
            pnl_pct         REAL,
            max_drawdown    REAL,
            avg_duration    REAL
        )
        """)

        # ── paper_trades: 모의 거래 로그 ─────────────────
        cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy        TEXT NOT NULL,      -- 'A' | 'B'
            side            TEXT NOT NULL,      -- 'long' | 'short'
            open_ts         TEXT NOT NULL,
            close_ts        TEXT,
            entry_price     REAL,
            exit_price      REAL,
            stop_loss       REAL,
            take_profit     REAL,
            pnl_pct         REAL,               -- 레버리지 미적용 단순 가격 변화율
            exit_reason     TEXT,               -- 'tp' | 'sl' | 'open'
            entry_rsi       REAL,
            entry_adx       REAL,
            entry_bb_pct_b  REAL,
            duration_min    REAL,
            reason          TEXT                -- 진입 이유 텍스트
        )
        """)

        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """기존 DB에 새 컬럼 추가 (없으면 추가, 있으면 무시)"""
        migrations = {
            "signals": [
                "weekday INTEGER",
                "hour_utc INTEGER",
                "atr_pct REAL",
                "ema_1h_trend TEXT",
                "rsi_1h REAL",
                "volume_ratio REAL",
                "funding_rate REAL",
                # 거래량 상세
                "vol_current REAL",
                "vol_sma20 REAL",
                "vol_spike INTEGER",
                "vol_dry INTEGER",
                "oi_current REAL",
                "oi_change_1h_pct REAL",
                "long_short_ratio REAL",
                "buy_vol_pct REAL",
                "vol_delta REAL",
                "cvd_slope_5 REAL",
                # 3개월 분석용
                "div_grade TEXT",
                "div_tf_count INTEGER",
                "div_strength INTEGER",
                "cvd_confirmed INTEGER",
                "bb_squeeze INTEGER",
                "rsi_4h REAL",
                "score_trend REAL",
                "score_breakout REAL",
            ],
            "trades": [
                "bb_pct_b REAL",
                "div_bars_between INTEGER",
                "early_direction INTEGER",
                "entry_adx REAL",
                "entry_1h_trend TEXT",
                "entry_4h_trend TEXT",
                "entry_rsi_1h REAL",
                "entry_vol_ratio REAL",
                "entry_funding REAL",
                "mfe_pct REAL",
                "mae_pct REAL",
                "mfe_r REAL",
                # 거래량 상세
                "vol_at_entry REAL",
                "vol_sma20_entry REAL",
                "vol_trend_5 REAL",
                "buy_vol_pct REAL",
                "oi_at_entry REAL",
                "oi_change_pct REAL",
                "long_short_ratio REAL",
                "vol_delta REAL",
                "cvd_slope_5 REAL",
                # 슬리피지
                "signal_price REAL",
                "slippage_pct REAL",
                # 청산 시점 지표
                "exit_ema_trend TEXT",
                "exit_macd_hist REAL",
                "exit_rsi REAL",
                "exit_atr REAL",
                "exit_vol_ratio REAL",
                # 3개월 분석용
                "div_grade TEXT",
                "div_tf_count INTEGER",
                "div_strength INTEGER",
                "cvd_confirmed INTEGER",
                "entry_bb_squeeze INTEGER",
                "entry_rsi_4h REAL",
                "score_trend REAL",
                "score_breakout REAL",
                "candle_body_pct REAL",
            ],
        }
        # shadow_trades는 신규 테이블이라 CREATE IF NOT EXISTS로 충분 (ALTER 불필요)
        for table, columns in migrations.items():
            existing = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for col_def in columns:
                col_name = col_def.split()[0]
                if col_name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                    logger.info(f"DB 마이그레이션: {table}.{col_name} 추가")
        self.conn.commit()

    # ── 신호 기록 ─────────────────────────────────────────
    def log_signal(
        self,
        price: float,
        ema_fast: float,
        ema_slow: float,
        macd_hist: float,
        macd_prev: float,
        rsi: float,
        atr: float,
        score: float,
        direction: str,
        fired: bool,
        reasons: list,
        ema_1h_trend: str = "unknown",
        rsi_1h: float = 0.0,
        volume_ratio: float = 0.0,
        funding_rate: float = 0.0,
        # 거래량 심화
        vol_current: float = 0.0,
        vol_sma20: float = 0.0,
        vol_spike: bool = False,
        vol_dry: bool = False,
        oi_current: float = 0.0,
        oi_change_1h_pct: float = 0.0,
        long_short_ratio: float = 0.0,
        buy_vol_pct: float = 0.0,
        vol_delta: float = 0.0,
        cvd_slope_5: float = 0.0,
        # 3개월 분석용
        div_grade: str = "none",
        div_tf_count: int = 0,
        div_strength: int = 0,
        cvd_confirmed: bool = False,
        bb_squeeze: bool = False,
        rsi_4h: float = 0.0,
        score_trend: float = 0.0,
        score_breakout: float = 0.0,
    ):
        if macd_prev < 0 and macd_hist > 0:
            macd_state = "bull_cross"
        elif macd_prev > 0 and macd_hist < 0:
            macd_state = "bear_cross"
        elif macd_hist > 0 and macd_hist > macd_prev:
            macd_state = "improving_pos"
        elif macd_hist > 0:
            macd_state = "positive"
        elif macd_hist < 0 and macd_hist > macd_prev:
            macd_state = "improving_neg"
        else:
            macd_state = "negative"

        now = datetime.now(timezone.utc)
        try:
            self.conn.execute("""
            INSERT INTO signals
                (ts, weekday, hour_utc, price,
                 ema_fast, ema_slow, ema_trend, macd_hist, macd_prev, macd_state,
                 rsi, atr, atr_pct,
                 ema_1h_trend, rsi_1h, volume_ratio, funding_rate,
                 score, direction, fired, reasons,
                 vol_current, vol_sma20, vol_spike, vol_dry,
                 oi_current, oi_change_1h_pct, long_short_ratio,
                 buy_vol_pct, vol_delta, cvd_slope_5,
                 div_grade, div_tf_count, div_strength, cvd_confirmed,
                 bb_squeeze, rsi_4h, score_trend, score_breakout)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now.isoformat(), now.weekday(), now.hour, price,
                ema_fast, ema_slow,
                "above" if ema_fast > ema_slow else "below",
                macd_hist, macd_prev, macd_state,
                rsi, atr, atr / price if price else 0,
                ema_1h_trend, rsi_1h, volume_ratio, funding_rate,
                score, direction, int(fired),
                json.dumps(reasons),
                vol_current, vol_sma20, int(vol_spike), int(vol_dry),
                oi_current, oi_change_1h_pct, long_short_ratio,
                buy_vol_pct, vol_delta, cvd_slope_5,
                div_grade, div_tf_count, div_strength, int(cvd_confirmed),
                int(bb_squeeze), rsi_4h, score_trend, score_breakout,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"신호 로그 저장 실패: {e}")

    # ── 트레이드 오픈 ─────────────────────────────────────
    def log_trade_open(
        self,
        trade_id: str,
        side: str,
        entry_price: float,
        amount: float,
        stop_loss: float,
        take_profit: float,
        leverage: int,
        ema_fast: float,
        ema_slow: float,
        macd_hist: float,
        macd_prev: float,
        rsi: float,
        atr: float,
        score: float,
        reasons: list,
        ema_1h_trend: str = "unknown",
        ema_4h_trend: str = "unknown",
        rsi_1h: float = 0.0,
        volume_ratio: float = 0.0,
        funding_rate: float = 0.0,
        # 거래량 심화
        vol_current: float = 0.0,
        vol_sma20: float = 0.0,
        vol_trend_5: float = 0.0,
        buy_vol_pct: float = 0.0,
        oi_current: float = 0.0,
        oi_change_pct: float = 0.0,
        long_short_ratio: float = 0.0,
        vol_delta: float = 0.0,
        cvd_slope_5: float = 0.0,
        signal_price: float = 0.0,
        # 3개월 분석용
        div_grade: str = "none",
        div_tf_count: int = 0,
        div_strength: int = 0,
        cvd_confirmed: bool = False,
        entry_bb_squeeze: bool = False,
        entry_rsi_4h: float = 0.0,
        score_trend: float = 0.0,
        score_breakout: float = 0.0,
        candle_body_pct: float = 0.0,
        bb_pct_b: float = 0.0,
        div_bars_between: int = 0,
        entry_adx: float = 0.0,
    ):
        now = datetime.now(timezone.utc)
        sl_pct = abs(entry_price - stop_loss) / entry_price if entry_price else 0
        slippage_pct = (entry_price - signal_price) / signal_price if signal_price > 0 else 0.0
        try:
            self.conn.execute("""
            INSERT OR REPLACE INTO trades
                (id, open_ts, side, entry_price, amount, stop_loss, take_profit,
                 sl_pct, leverage,
                 entry_ema_fast, entry_ema_slow, entry_macd_hist, entry_macd_prev,
                 entry_rsi, entry_atr, entry_score, entry_reasons,
                 entry_1h_trend, entry_4h_trend, entry_rsi_1h,
                 entry_vol_ratio, entry_funding,
                 weekday, hour_utc,
                 vol_at_entry, vol_sma20_entry, vol_trend_5,
                 buy_vol_pct, oi_at_entry, oi_change_pct,
                 long_short_ratio, vol_delta, cvd_slope_5,
                 signal_price, slippage_pct,
                 div_grade, div_tf_count, div_strength, cvd_confirmed,
                 entry_bb_squeeze, entry_rsi_4h,
                 score_trend, score_breakout, candle_body_pct,
                 bb_pct_b, div_bars_between, entry_adx)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                trade_id, now.isoformat(), side,
                entry_price, amount, stop_loss, take_profit,
                sl_pct, leverage,
                ema_fast, ema_slow, macd_hist, macd_prev,
                rsi, atr, score,
                json.dumps(reasons),
                ema_1h_trend, ema_4h_trend, rsi_1h,
                volume_ratio, funding_rate,
                now.weekday(), now.hour,
                vol_current, vol_sma20, vol_trend_5,
                buy_vol_pct, oi_current, oi_change_pct,
                long_short_ratio, vol_delta, cvd_slope_5,
                signal_price, slippage_pct,
                div_grade, div_tf_count, div_strength, int(cvd_confirmed),
                int(entry_bb_squeeze), entry_rsi_4h,
                score_trend, score_breakout, candle_body_pct,
                bb_pct_b, div_bars_between, entry_adx,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"트레이드 오픈 로그 저장 실패: {e}")

    # ── Shadow Trade 기록 ────────────────────────────────────
    def log_shadow_trade(
        self,
        direction: str,
        score: float,
        signal_price: float,
        stop_loss: float,
        take_profit: float,
        reasons: list,
        active_side: str,
        active_entry: float,
        active_pnl_pct: float,
    ) -> int:
        """포지션 보유 중 무시된 반대 시그널을 가상 트레이드로 기록. row id 반환."""
        now = datetime.now(timezone.utc)
        sl_pct = abs(signal_price - stop_loss) / signal_price if signal_price > 0 else 0
        try:
            cur = self.conn.execute("""
            INSERT INTO shadow_trades
                (ts, direction, score, signal_price, stop_loss, take_profit, sl_pct,
                 reasons, active_side, active_entry, active_pnl_pct, outcome)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,'open')
            """, (
                now.isoformat(), direction, score, signal_price,
                stop_loss, take_profit, sl_pct,
                json.dumps(reasons),
                active_side, active_entry, active_pnl_pct,
            ))
            self.conn.commit()
            return cur.lastrowid
        except Exception as e:
            logger.error(f"Shadow trade 저장 실패: {e}")
            return -1

    def update_shadow_trade(
        self,
        row_id: int,
        outcome: str,          # 'tp' | 'sl'
        exit_price: float,
        virtual_pnl_pct: float,
        duration_min: float,
    ):
        """Shadow trade 가상 결과 업데이트 (SL/TP 터치 시)."""
        try:
            self.conn.execute("""
            UPDATE shadow_trades SET
                outcome=?, exit_price=?, exit_ts=?,
                virtual_pnl_pct=?, duration_min=?
            WHERE id=?
            """, (
                outcome, exit_price,
                datetime.now(timezone.utc).isoformat(),
                virtual_pnl_pct, duration_min,
                row_id,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Shadow trade 업데이트 실패: {e}")

    # ── 거래량 프로파일 스냅샷 (5분마다) ─────────────────────
    def log_volume_profile(
        self,
        price: float,
        vol_stats: dict,
        vol_delta: dict,
        oi: float,
        oi_change_pct: float,
        long_short_ratio: float,
    ):
        now = datetime.now(timezone.utc)
        try:
            self.conn.execute("""
            INSERT INTO volume_profile
                (ts, price,
                 vol_current, vol_sma20, vol_ratio, vol_spike, vol_dry,
                 vol_trend_5, buy_vol_pct,
                 vol_delta, vol_delta_cum, cvd_slope_5,
                 oi, oi_change_pct, long_short_ratio)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now.isoformat(), price,
                vol_stats.get("vol_current", 0),
                vol_stats.get("vol_sma20", 0),
                vol_stats.get("vol_ratio", 0),
                int(vol_stats.get("vol_spike", False)),
                int(vol_stats.get("vol_dry", False)),
                vol_stats.get("vol_trend_5", 0),
                vol_stats.get("buy_vol_pct", 0),
                vol_delta.get("vol_delta", 0),
                vol_delta.get("vol_delta_cumulative", 0),
                vol_delta.get("cvd_slope_5", 0),
                oi, oi_change_pct, long_short_ratio,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"거래량 프로파일 저장 실패: {e}")

    # ── 진입 직후 첫 3봉 방향 업데이트 ─────────────────
    def update_early_direction(self, trade_id: str, direction_ok: int):
        """진입 후 3봉(45분) 지난 시점에 방향 일치 여부 기록 (1=맞음, 0=역행)"""
        try:
            self.conn.execute(
                "UPDATE trades SET early_direction=? WHERE id=?",
                (direction_ok, trade_id)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"early_direction 업데이트 실패: {e}")

    # ── MFE/MAE 업데이트 (포지션 모니터링 중 매 틱) ──────
    def update_excursion(self, trade_id: str, mfe_pct: float, mae_pct: float, mfe_r: float = 0.0):
        """포지션 보유 중 최대 유리/불리 이동 + R배수 업데이트"""
        try:
            self.conn.execute("""
            UPDATE trades SET mfe_pct=?, mae_pct=?, mfe_r=? WHERE id=?
            """, (mfe_pct, mae_pct, mfe_r, trade_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"MFE/MAE 업데이트 실패: {e}")

    # ── 트레이드 클로즈 ───────────────────────────────────
    def log_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        pnl_usdt: float,
        pnl_pct: float,
        exit_reason: str,
        duration_min: float,
        exit_ema_trend: str = "unknown",
        exit_macd_hist: float = 0.0,
        exit_rsi: float = 0.0,
        exit_atr: float = 0.0,
        exit_vol_ratio: float = 0.0,
    ):
        try:
            self.conn.execute("""
            UPDATE trades SET
                close_ts=?, exit_price=?, pnl_usdt=?, pnl_pct=?,
                exit_reason=?, duration_min=?,
                exit_ema_trend=?, exit_macd_hist=?, exit_rsi=?,
                exit_atr=?, exit_vol_ratio=?
            WHERE id=?
            """, (
                datetime.now(timezone.utc).isoformat(),
                exit_price, pnl_usdt, pnl_pct,
                exit_reason, duration_min,
                exit_ema_trend, exit_macd_hist, exit_rsi,
                exit_atr, exit_vol_ratio,
                trade_id,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"트레이드 클로즈 로그 저장 실패: {e}")

    # ── 일간 스냅샷 ───────────────────────────────────────
    def log_daily(
        self,
        date: str,
        start_balance: float,
        end_balance: float,
        stats: dict,
    ):
        trades = self.conn.execute(
            "SELECT duration_min FROM trades WHERE DATE(open_ts)=? AND close_ts IS NOT NULL",
            (date,)
        ).fetchall()
        avg_dur = sum(r[0] for r in trades if r[0]) / len(trades) if trades else 0
        try:
            self.conn.execute("""
            INSERT OR REPLACE INTO daily
                (date, start_balance, end_balance, total_trades, wins,
                 losses, pnl_usdt, pnl_pct, max_drawdown, avg_duration)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                date, start_balance, end_balance,
                stats.get("total", 0),
                stats.get("wins", 0),
                stats.get("losses", 0),
                stats.get("total_pnl_usdt", 0),
                stats.get("total_pnl_pct", 0),
                stats.get("max_drawdown", 0),
                avg_dur,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"일간 스냅샷 저장 실패: {e}")

    # ── DB → 메모리 복원 (재시작 시 trade_history 재구성) ───
    def load_trade_history(self) -> list:
        """
        DB의 완료된 트레이드를 Trade 객체 형태의 dict 리스트로 반환.
        재시작 후 order_mgr.trade_history에 적재해 누적 통계 유지.
        """
        try:
            rows = self.conn.execute("""
                SELECT side, entry_price, exit_price, amount,
                       pnl_usdt, pnl_pct, exit_reason, open_ts
                FROM trades
                WHERE close_ts IS NOT NULL
                ORDER BY open_ts ASC
            """).fetchall()
        except Exception as e:
            logger.error(f"거래 히스토리 로드 실패: {e}")
            return []

        history = []
        for r in rows:
            history.append({
                "side": r[0],
                "entry_price": r[1] or 0.0,
                "exit_price": r[2] or 0.0,
                "amount": r[3] or 0.0,
                "pnl": r[4] or 0.0,
                "pnl_pct": r[5] or 0.0,
                "exit_reason": r[6] or "",
                "open_ts": r[7] or "",
            })
        logger.info(f"📂 거래 히스토리 복원: {len(history)}건 (DB 기준)")
        return history

    # ── 누적 통계 조회 (DB 기반, 재시작해도 유지) ────────
    def get_cumulative_stats(self) -> dict:
        """
        DB에 기록된 전체 완료 트레이드 기준 누적 통계
        Returns: total, wins, losses, win_rate, profit_factor,
                 avg_win_pct, avg_loss_pct, total_pnl_usdt
        """
        try:
            rows = self.conn.execute(
                "SELECT pnl_usdt, pnl_pct FROM trades WHERE close_ts IS NOT NULL"
            ).fetchall()
        except Exception:
            return {}

        if not rows:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0}

        wins = [r for r in rows if r[0] is not None and r[0] > 0]
        losses = [r for r in rows if r[0] is not None and r[0] <= 0]
        total = len(rows)

        avg_win_pct = sum(r[1] for r in wins) / len(wins) if wins else 0.0
        avg_loss_pct = sum(r[1] for r in losses) / len(losses) if losses else 0.0
        profit_factor = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct else float("inf")
        total_pnl = sum(r[0] for r in rows if r[0] is not None)

        return {
            "total": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / total,
            "profit_factor": profit_factor,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "total_pnl_usdt": total_pnl,
        }

    # ── R:R 최적화 분석 (3개월 데이터 기반) ─────────────────
    def get_rr_optimization_stats(self, div_grade: str = None) -> dict:
        """
        mfe_r 데이터로 손익비(R:R) 최적값 분석

        각 R 레벨(1.5 ~ 8.0)에 대해 계산:
          - hit_rate  : 해당 R에 도달한 트레이드 비율
          - win_rate  : TP를 그 R에 뒀을 때 승률 (hit_rate = 새로운 승률)
          - profit_factor : 해당 R 기준 수익 팩터 (win*R / loss*1)
          - expectancy: 트레이드당 기대값 (R 단위)

        div_grade 지정 시 해당 등급만 분석 (예: 'A', 'B')
        """
        where = "WHERE close_ts IS NOT NULL AND mfe_r IS NOT NULL"
        params: list = []
        if div_grade:
            where += " AND div_grade=?"
            params.append(div_grade)

        try:
            rows = self.conn.execute(
                f"SELECT mfe_r, mae_pct FROM trades {where}", params
            ).fetchall()
        except Exception as e:
            logger.error(f"R:R 분석 조회 실패: {e}")
            return {}

        if not rows:
            return {"total": 0, "message": "데이터 없음"}

        total = len(rows)
        mfe_rs = [r[0] for r in rows if r[0] is not None]

        r_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
        analysis = {}
        for r in r_levels:
            hits = sum(1 for v in mfe_rs if v >= r)
            hit_rate = hits / total if total > 0 else 0
            # 해당 R에서 TP 걸었을 때:
            # 승: hit_rate 확률로 R배 수익
            # 패: (1-hit_rate) 확률로 -1R 손실
            expectancy = hit_rate * r - (1 - hit_rate) * 1.0
            pf = (hit_rate * r) / ((1 - hit_rate) * 1.0) if (1 - hit_rate) > 0 else float("inf")
            analysis[f"1:{r}"] = {
                "r": r,
                "hit_trades": hits,
                "hit_rate": round(hit_rate, 4),
                "profit_factor": round(pf, 3),
                "expectancy_r": round(expectancy, 4),  # 트레이드당 기대 R
                "breakeven_winrate": round(1 / (1 + r), 4),  # 손익분기 승률
            }

        # 최적 R 찾기 (expectancy 최대)
        best_r = max(analysis, key=lambda k: analysis[k]["expectancy_r"])

        result = {
            "total_trades": total,
            "avg_mfe_r": round(sum(mfe_rs) / len(mfe_rs), 3) if mfe_rs else 0,
            "max_mfe_r": round(max(mfe_rs), 3) if mfe_rs else 0,
            "grade_filter": div_grade or "전체",
            "best_rr": best_r,
            "best_expectancy": analysis[best_r]["expectancy_r"],
            "by_r_level": analysis,
        }
        logger.info(
            f"📊 R:R 최적화 분석 | 총 {total}건 | 최적={best_r} "
            f"(기대값={analysis[best_r]['expectancy_r']:+.3f}R)"
        )
        return result

    def get_rr_summary_log(self) -> str:
        """등급별 R:R 최적화 요약 (텔레그램/로그 출력용)"""
        lines = ["📊 R:R 최적화 분석 리포트"]
        for grade in ["A", "B", "C", "D", "E", "전체"]:
            g = None if grade == "전체" else grade
            stats = self.get_rr_optimization_stats(div_grade=g)
            if stats.get("total_trades", 0) < 5:
                continue
            lines.append(
                f"\n[{grade}급] {stats['total_trades']}건 | "
                f"평균MFE={stats['avg_mfe_r']}R | "
                f"최적={stats['best_rr']} "
                f"(기대값={stats['best_expectancy']:+.3f}R)"
            )
            # 주요 R 레벨 히트율 출력
            for key in ["1:3.0", "1:4.0", "1:5.0", "1:6.0"]:
                d = stats["by_r_level"].get(key, {})
                if d:
                    lines.append(
                        f"  {key}: 히트율={d['hit_rate']:.1%} | "
                        f"PF={d['profit_factor']:.2f} | "
                        f"기대={d['expectancy_r']:+.3f}R"
                    )
        return "\n".join(lines)

    # ── 모의 거래 기록 ────────────────────────────────────
    def log_paper_trade_open(
        self,
        strategy: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        entry_rsi: float = 0.0,
        entry_adx: float = 0.0,
        entry_bb_pct_b: float = 0.5,
        reason: str = "",
    ) -> int:
        """모의 진입 기록, 생성된 row id 반환"""
        now = datetime.now(timezone.utc)
        try:
            cur = self.conn.execute("""
            INSERT INTO paper_trades
                (strategy, side, open_ts, entry_price, stop_loss, take_profit,
                 entry_rsi, entry_adx, entry_bb_pct_b, reason)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                strategy, side, now.isoformat(),
                entry_price, stop_loss, take_profit,
                entry_rsi, entry_adx, entry_bb_pct_b, reason,
            ))
            self.conn.commit()
            return cur.lastrowid
        except Exception as e:
            logger.error(f"모의 거래 진입 저장 실패: {e}")
            return -1

    def log_paper_trade_close(
        self,
        row_id: int,
        exit_price: float,
        pnl_pct: float,
        exit_reason: str,
        duration_min: float,
    ):
        """모의 청산 업데이트"""
        now = datetime.now(timezone.utc)
        try:
            self.conn.execute("""
            UPDATE paper_trades
            SET close_ts=?, exit_price=?, pnl_pct=?, exit_reason=?, duration_min=?
            WHERE id=?
            """, (
                now.isoformat(), exit_price, pnl_pct,
                exit_reason, duration_min, row_id,
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"모의 거래 청산 업데이트 실패: {e}")

    def get_paper_stats(self, strategy: str = None) -> dict:
        """모의 거래 승률/손익비 조회"""
        where = "WHERE close_ts IS NOT NULL"
        params: list = []
        if strategy:
            where += " AND strategy=?"
            params.append(strategy)
        try:
            rows = self.conn.execute(
                f"SELECT pnl_pct, exit_reason FROM paper_trades {where}", params
            ).fetchall()
        except Exception:
            return {}
        if not rows:
            return {"total": 0}
        wins = [r for r in rows if r[0] is not None and r[0] > 0]
        losses = [r for r in rows if r[0] is not None and r[0] <= 0]
        total = len(rows)
        avg_win = sum(r[0] for r in wins) / len(wins) if wins else 0.0
        avg_loss = sum(r[0] for r in losses) / len(losses) if losses else 0.0
        pf = abs(avg_win / avg_loss) if avg_loss else float("inf")
        return {
            "strategy": strategy or "전체",
            "total": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / total, 4) if total else 0,
            "avg_win_pct": round(avg_win, 4),
            "avg_loss_pct": round(avg_loss, 4),
            "profit_factor": round(pf, 3),
        }

    def close(self):
        self.conn.close()
