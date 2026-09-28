"""Indicator and strategy tests."""
from __future__ import annotations

import pytest

from bot.core.models import Side
from bot.strategies import available, build
from bot.strategies.indicators import atr, ema, rolling_max, rolling_min, rsi, sma, stdev, zscore
from tests.conftest import candle_series


class TestIndicators:
    def test_sma_of_a_known_series(self):
        assert sma([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]

    def test_sma_length(self):
        values = list(range(10))
        assert len(sma(values, 4)) == len(values) - 4 + 1

    def test_sma_rejects_bad_period(self):
        with pytest.raises(ValueError):
            sma([1, 2, 3], 0)

    def test_ema_seeds_from_the_sma(self):
        result = ema([1, 2, 3, 4, 5], 3)
        assert result[0] == pytest.approx(2.0)

    def test_ema_follows_a_trend_upward(self):
        result = ema(list(range(1, 20)), 5)
        assert result[-1] > result[0]

    def test_rsi_is_100_on_a_pure_uptrend(self):
        assert rsi([float(i) for i in range(1, 40)], 14)[-1] == pytest.approx(100.0)

    def test_rsi_is_low_on_a_downtrend(self):
        values = [float(40 - i) for i in range(40)]
        assert rsi(values, 14)[-1] < 10.0

    def test_rsi_stays_inside_bounds(self):
        values = [1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.05, 1.25, 0.95, 1.15,
                  1.35, 0.85, 1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.05, 1.25]
        for value in rsi(values, 14):
            assert 0.0 <= value <= 100.0

    def test_atr_is_positive_for_a_volatile_series(self):
        highs = [1.1010, 1.1030, 1.1000, 1.1050, 1.1020]
        lows = [1.0990, 1.1000, 1.0970, 1.1010, 1.0990]
        closes = [1.1000, 1.1020, 1.0980, 1.1040, 1.1000]
        assert atr(highs, lows, closes, 3)[-1] > 0

    def test_atr_is_flat_for_a_flat_series(self):
        highs = lows = closes = [1.10] * 30
        assert atr(highs, lows, closes, 14)[-1] == pytest.approx(0.0)

    def test_rolling_max_over_a_window(self):
        assert rolling_max([1, 3, 2, 5, 4], 3) == [3, 5, 5]

    def test_rolling_min_over_a_window(self):
        assert rolling_min([1, 3, 2, 5, 4], 3) == [1, 2, 2]

    def test_stdev_of_a_constant_is_zero(self):
        assert stdev([5.0, 5.0, 5.0]) == 0.0

    def test_zscore_is_zero_for_a_flat_series(self):
        assert all(v == 0.0 for v in zscore([1.0] * 20, 10))


class TestRegistry:
    def test_all_strategies_are_registered(self):
        assert set(available()) == {"ma_crossover", "rsi_reversion", "breakout"}

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError):
            build("does_not_exist")

    def test_build_passes_params(self):
        strategy = build("ma_crossover", fast=5, slow=20)
        assert strategy.params["fast"] == 5
        assert strategy.params["slow"] == 20
        assert strategy.fast == 5
        assert strategy.slow == 20


class TestMovingAverageCrossover:
    def test_no_signal_during_warmup(self):
        strategy = build("ma_crossover")
        series = candle_series("EUR/USD", [1.10] * 10)
        assert all(strategy.on_candle(series[: i + 1]) is None for i in range(len(series)))

    def test_signals_long_on_a_golden_cross(self):
        strategy = build("ma_crossover", fast=3, slow=10)
        # flat, then a steady rally -> the first cross is golden
        closes = [1.10] * 12 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16]
        series = candle_series("EUR/USD", closes)
        signals = [s for s in (strategy.on_candle(series[: i + 1]) for i in range(len(series))) if s]
        assert signals
        assert signals[0].side is Side.BUY
        assert signals[0].stop_distance is not None
        assert signals[0].reason

    def test_signals_short_on_a_death_cross(self):
        strategy = build("ma_crossover", fast=3, slow=10)
        # flat, then a steady slide -> the first cross is a death cross
        closes = [1.10] * 12 + [1.09, 1.08, 1.07, 1.06, 1.05]
        series = candle_series("EUR/USD", closes)
        signals = [s for s in (strategy.on_candle(series[: i + 1]) for i in range(len(series))) if s]
        assert signals
        assert signals[0].side is Side.SELL

    def test_rejects_inverted_periods(self):
        with pytest.raises(ValueError):
            build("ma_crossover", fast=50, slow=10)


class TestRSIMeanReversion:
    def test_buys_when_rsi_leaves_oversold(self):
        strategy = build("rsi_reversion", period=5, oversold=30, overbought=70)
        # a long slide then a single bounce
        closes = [1.20 - 0.01 * i for i in range(30)] + [1.09, 1.11]
        series = candle_series("EUR/USD", closes)
        signals = [s for s in (strategy.on_candle(series[: i + 1]) for i in range(len(series))) if s]
        assert signals
        assert signals[0].side is Side.BUY

    def test_stop_distance_scales_with_atr(self):
        strategy = build("rsi_reversion", period=5, atr_mult=2.0)
        quiet = candle_series("EUR/USD", [1.10 + 0.0001 * (i % 3) for i in range(40)])
        wild = candle_series("EUR/USD", [1.10 + 0.01 * (i % 3) for i in range(40)])
        quiet_signal = strategy.on_candle(quiet)
        wild_signal = strategy.on_candle(wild)
        if quiet_signal and wild_signal:
            assert wild_signal.stop_distance > quiet_signal.stop_distance

    def test_strength_is_bounded(self):
        strategy = build("rsi_reversion", period=5)
        closes = [1.20 - 0.01 * i for i in range(40)]
        series = candle_series("EUR/USD", closes)
        for i in range(len(series)):
            signal = strategy.on_candle(series[: i + 1])
            if signal:
                assert 0.25 <= signal.strength <= 1.0


class TestBreakout:
    def test_signals_long_above_the_lookback_high(self):
        strategy = build("breakout", lookback=5)
        closes = [1.10, 1.11, 1.09, 1.12, 1.08, 1.10, 1.15]
        series = candle_series("EUR/USD", closes, wick=0.0005)
        signals = [s for s in (strategy.on_candle(series[: i + 1]) for i in range(len(series))) if s]
        assert signals
        assert signals[0].side is Side.BUY
        assert "break above" in signals[0].reason

    def test_signals_short_below_the_lookback_low(self):
        strategy = build("breakout", lookback=5)
        closes = [1.10, 1.09, 1.11, 1.08, 1.12, 1.10, 1.05]
        series = candle_series("EUR/USD", closes, wick=0.0005)
        signals = [s for s in (strategy.on_candle(series[: i + 1]) for i in range(len(series))) if s]
        assert signals
        assert signals[0].side is Side.SELL

    def test_ignores_the_current_bar_when_measuring_the_range(self):
        """The breakout level must come from prior bars only."""
        strategy = build("breakout", lookback=5)
        # the last bar makes a new high but the prior range was lower
        closes = [1.10, 1.10, 1.10, 1.10, 1.10, 1.10, 1.20]
        series = candle_series("EUR/USD", closes, wick=0.0005)
        assert strategy.on_candle(series) is not None
        # a flat series never breaks out
        flat = candle_series("EUR/USD", [1.10] * 20, wick=0.0)
        assert strategy.on_candle(flat) is None


class TestStrategyIsStateless:
    def test_same_history_gives_the_same_answer(self):
        """Strategies must be pure functions of history - no hidden state."""
        strategy = build("ma_crossover", fast=3, slow=10)
        closes = [1.10] * 12 + [1.09, 1.08, 1.07, 1.06] + [1.07, 1.09, 1.11, 1.13, 1.15]
        series = candle_series("EUR/USD", closes)
        first = strategy.on_candle(series)
        second = strategy.on_candle(series)
        assert (first is None) == (second is None)
        if first is not None:
            assert first.side == second.side
            assert first.reason == second.reason
