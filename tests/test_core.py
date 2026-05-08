"""기본 유닛 테스트 - 인디케이터 & 리스크 로직 검증"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy.base import ema, sma, rsi, macd, bollinger_bands, atr, fibonacci_levels


# ── 인디케이터 테스트 ──

class TestIndicators:

    def setup_method(self):
        """테스트용 가격 데이터 생성"""
        np.random.seed(42)
        self.prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        self.highs = self.prices + np.abs(np.random.randn(200) * 0.3)
        self.lows = self.prices - np.abs(np.random.randn(200) * 0.3)

    def test_ema_length(self):
        result = ema(self.prices, 9)
        assert len(result) == len(self.prices)

    def test_ema_smoothing(self):
        """EMA는 원본보다 변동이 작아야 함"""
        result = ema(self.prices, 50)
        assert np.std(np.diff(result[50:])) < np.std(np.diff(self.prices[50:]))

    def test_sma_matches_manual(self):
        result = sma(self.prices, 5)
        manual = np.mean(self.prices[:5])
        assert abs(result[4] - manual) < 1e-10

    def test_rsi_range(self):
        result = rsi(self.prices, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_macd_components(self):
        line, signal, hist = macd(self.prices, 12, 26, 9)
        assert len(line) == len(self.prices)
        # 히스토그램 = MACD - 시그널
        np.testing.assert_allclose(hist, line - signal, rtol=1e-10)

    def test_bollinger_bands_order(self):
        upper, middle, lower = bollinger_bands(self.prices, 20, 2.0)
        valid_idx = ~np.isnan(upper)
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_atr_positive(self):
        result = atr(self.highs, self.lows, self.prices, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_fibonacci_levels(self):
        levels = fibonacci_levels(100.0, 50.0)
        assert levels[0.0] == 100.0
        assert levels[1.0] == 50.0
        assert abs(levels[0.5] - 75.0) < 1e-10
        assert abs(levels[0.618] - 69.1) < 0.1


# ── 포지션 사이징 테스트 ──

class TestPositionSizing:

    def test_basic_sizing(self):
        """고정 비율법 기본 검증"""
        from config import Config
        from risk.risk_manager import PositionSizer

        config = Config(
            risk={"risk_per_trade_pct": 0.02, "max_position_pct": 0.3}
        )
        sizer = PositionSizer(config)

        # 잔고 10000, 진입가 70000, SL 68000 (2.86%), 레버리지 10
        amount = sizer.calculate_size(
            balance=10000,
            entry_price=70000,
            stop_loss=68000,
            leverage=10,
        )

        # 리스크 금액 = 10000 * 0.02 = 200
        # SL폭 = 2000/70000 = 2.86%
        # 포지션 가치 = 200 / (0.0286 * 10) = 699
        # BTC 수량 = 699 / 70000 ≈ 0.01
        assert amount > 0
        assert amount <= 10000 * 0.3 / 70000  # 최대 포지션 이하

    def test_zero_stop_returns_zero(self):
        from config import Config
        from risk.risk_manager import PositionSizer

        config = Config(risk={"risk_per_trade_pct": 0.02, "max_position_pct": 0.3})
        sizer = PositionSizer(config)
        assert sizer.calculate_size(10000, 70000, 0, 10) == 0.0


# ── 드로다운 가드 테스트 ──

class TestDrawdownGuard:

    def test_daily_loss_limit(self):
        from config import Config
        from risk.risk_manager import DrawdownGuard

        config = Config(
            risk={"daily_max_loss_pct": 0.05, "daily_max_trades": 10, "consecutive_loss_pause": 3, "cooldown_minutes": 60}
        )
        guard = DrawdownGuard(config)
        guard.reset_daily(10000)

        # 5% 이상 손실 시 거래 불가
        can, reason = guard.can_trade(9400)  # -6%
        assert can is False
        assert "일간 손실" in reason

    def test_consecutive_loss_cooldown(self):
        from config import Config
        from risk.risk_manager import DrawdownGuard

        config = Config(
            risk={"daily_max_loss_pct": 0.05, "daily_max_trades": 10, "consecutive_loss_pause": 3, "cooldown_minutes": 60}
        )
        guard = DrawdownGuard(config)
        guard.reset_daily(10000)

        guard.record_trade(-50)
        guard.record_trade(-30)
        guard.record_trade(-40)  # 3연패 → 쿨다운

        can, reason = guard.can_trade(9880)
        assert can is False
        assert "쿨다운" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
