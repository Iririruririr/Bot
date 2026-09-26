"""Grid ladder, backtest, storage and config tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.backtest.runner import run_backtest
from bot.backtest.stats import BARS_PER_YEAR, compute_stats
from bot.brokers.paper import BrokerConfig, PaperBroker
from bot.config import BotConfig, has_oanda_credentials
from bot.core.models import Side, Trade
from bot.core.risk import RiskConfig
from bot.data.feed import CsvFeed, SimConfig, SimulatedFeed, generate_csv
from bot.data.storage import SQLiteStore
from bot.scaling.engine import ScalingConfig
from bot.scaling.ladder import GridLadder, LadderConfig
from bot.strategies import build
from tests.conftest import candle_series, make_candle

START = datetime(2024, 1, 2, tzinfo=timezone.utc)


@pytest.fixture
def broker():
    return PaperBroker(BrokerConfig(starting_cash=100_000.0, spread_pips=1.0, slippage_pips=0.2))


# --------------------------------------------------------------------------- #
# grid ladder
# --------------------------------------------------------------------------- #


class TestLadderConfig:
    def test_rejects_bad_direction(self):
        with pytest.raises(ValueError):
            LadderConfig(centre_price=1.08, direction="up")

    def test_rejects_bad_levels(self):
        with pytest.raises(ValueError):
            LadderConfig(centre_price=1.08, levels=0)

    def test_rejects_zero_spacing(self):
        with pytest.raises(ValueError):
            LadderConfig(centre_price=1.08, spacing_pips=0)


class TestGridLadder:
    def make_ladder(self, broker, **kwargs):
        """Ladder with 3 buy rungs unless the test overrides them."""
        params = dict(centre_price=1.0800, levels=3, spacing_pips=25.0,
                      qty_per_level=10_000, take_profit_pips=20.0, direction="buy")
        params.update(kwargs)
        config = LadderConfig(**params)
        return GridLadder(config, broker, symbol="EUR/USD"), config

    def test_places_buy_rungs_below_centre(self, broker):
        ladder, _ = self.make_ladder(broker, direction="buy")
        ladder.build()
        prices = [rung.price for rung in ladder.rungs]
        assert prices == sorted(prices, reverse=True)
        assert all(price < 1.0800 for price in prices)
        assert len(broker.working_orders()) == 3

    def test_places_sell_rungs_above_centre(self, broker):
        ladder, _ = self.make_ladder(broker, direction="sell")
        ladder.build()
        assert all(rung.price > 1.0800 for rung in ladder.rungs)
        assert all(rung.side is Side.SELL for rung in ladder.rungs)

    def test_both_sides_gives_two_rungs_per_level(self, broker):
        ladder, _ = self.make_ladder(broker, direction="both")
        ladder.build()
        assert len(ladder.rungs) == 6

    def test_rung_spacing_is_in_pips(self, broker):
        ladder, _ = self.make_ladder(broker, levels=3, spacing_pips=25.0)
        ladder.build()
        gaps = [abs(ladder.rungs[i].price - ladder.rungs[i + 1].price) for i in range(2)]
        assert gaps[0] == pytest.approx(0.0025)   # 25 pips
        assert gaps[1] == pytest.approx(0.0025)

    def test_jpy_pair_uses_jpy_pips(self, broker):
        config = LadderConfig(centre_price=150.00, levels=2, spacing_pips=50.0,
                              qty_per_level=10_000, take_profit_pips=30.0, direction="buy")
        ladder = GridLadder(config, broker, symbol="USD/JPY")
        ladder.build()
        gaps = [abs(ladder.rungs[0].price - ladder.rungs[1].price)]
        assert gaps[0] == pytest.approx(0.50)   # 50 JPY pips

    def test_fill_arms_a_take_profit(self, broker):
        ladder, _ = self.make_ladder(broker, direction="buy")
        ladder.build()
        # trade down through the first rung
        broker.on_candle(make_candle("EUR/USD", 1.0770, index=1, low=1.0760, high=1.0780))
        ladder.on_candle(make_candle("EUR/USD", 1.0770, index=1, low=1.0760, high=1.0780))
        assert ladder.stats["fills"] == 1
        rung = ladder.rungs[0]
        assert rung.filled_at is not None
        assert rung.target is not None
        assert rung.target.reduce_only is True

    def test_target_sits_the_configured_distance_away(self, broker):
        ladder, _ = self.make_ladder(broker, direction="buy", take_profit_pips=20.0)
        ladder.build()
        candle = make_candle("EUR/USD", 1.0770, index=1, low=1.0760, high=1.0780)
        broker.on_candle(candle)
        ladder.on_candle(candle)
        target = ladder.rungs[0].target
        assert target.limit_price == pytest.approx(ladder.rungs[0].price + 0.0020)

    def test_full_cycle_fills_then_closes_for_a_profit(self, broker):
        ladder, _ = self.make_ladder(broker, direction="buy")
        ladder.build()
        # down through the rung, then back up through the target
        down = make_candle("EUR/USD", 1.0770, index=1, low=1.0760, high=1.0780)
        up = make_candle("EUR/USD", 1.0800, index=2, low=1.0770, high=1.0810)
        for candle in (down, up):
            broker.on_candle(candle)
            ladder.on_candle(candle)
        assert ladder.stats["targets_hit"] == 1
        assert broker.closed_trades()
        assert broker.closed_trades()[0].net_pnl > 0

    def test_rearm_places_a_new_entry_order(self, broker):
        ladder, _ = self.make_ladder(broker, direction="buy", rearm=True)
        ladder.build()
        down = make_candle("EUR/USD", 1.0770, index=1, low=1.0760, high=1.0780)
        up = make_candle("EUR/USD", 1.0800, index=2, low=1.0770, high=1.0810)
        for candle in (down, up):
            broker.on_candle(candle)
            ladder.on_candle(candle)
        assert ladder.stats["rearmed"] == 1
        assert any(rung.state == "open" for rung in ladder.rungs)

    def test_hard_stop_flattens_everything(self, broker):
        ladder, _ = self.make_ladder(broker, direction="buy", stop_loss_pips=50.0)
        ladder.build()
        crash = make_candle("EUR/USD", 1.0700, index=1, low=1.0690, high=1.0710)
        broker.on_candle(crash)
        ladder.on_candle(crash)
        # 100 pips from the centre breaches the 50 pip hard stop
        assert broker.working_orders() == []
        assert not broker.positions()

    def test_summary_reports_rung_states(self, broker):
        ladder, _ = self.make_ladder(broker)
        ladder.build()
        summary = ladder.summary()
        assert summary["rungs"] == 3
        assert summary["open"] == 3
        assert summary["filled"] == 0


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #


class TestBacktest:
    def test_runs_end_to_end(self):
        feed = SimulatedFeed(SimConfig(symbol="EUR/USD", bars=500, seed=3))
        result = run_backtest(
            feed=feed,
            strategy=build("ma_crossover", fast=5, slow=20),
            scaling=ScalingConfig.pyramid(tranches=2, step_r=1.0),
            risk=RiskConfig(risk_per_trade_pct=1.0),
        )
        assert result.stats.trades >= 0
        assert len(result.equity_curve) == 500
        assert result.meta["strategy"] == "ma_crossover"
        assert result.meta["bars"] == 500

    def test_equity_curve_starts_at_the_starting_cash(self):
        feed = SimulatedFeed(SimConfig(symbol="EUR/USD", bars=200, seed=5))
        result = run_backtest(feed=feed, strategy=build("rsi_reversion"))
        assert result.equity_curve[0][1] == pytest.approx(100_000.0)

    def test_scaling_stats_are_reported(self):
        feed = SimulatedFeed(SimConfig(symbol="EUR/USD", bars=800, seed=11))
        result = run_backtest(
            feed=feed,
            strategy=build("ma_crossover", fast=5, slow=20),
            scaling=ScalingConfig.pyramid(tranches=3, step_r=1.0),
        )
        assert set(result.scaling_stats) >= {"adds", "scale_outs", "breakevens"}

    def test_no_scaling_config_means_no_adds(self):
        feed = SimulatedFeed(SimConfig(symbol="EUR/USD", bars=600, seed=13))
        result = run_backtest(
            feed=feed, strategy=build("ma_crossover", fast=5, slow=20), scaling=ScalingConfig()
        )
        assert result.scaling_stats["adds"] == 0
        assert result.scaling_stats["scale_outs"] == 0

    def test_deterministic_for_a_fixed_seed(self):
        def run():
            feed = SimulatedFeed(SimConfig(symbol="EUR/USD", bars=400, seed=99))
            return run_backtest(feed=feed, strategy=build("ma_crossover", fast=5, slow=20)).stats.net_pnl

        assert run() == run()

    def test_costs_reduce_returns(self):
        cheap = run_backtest(
            feed=SimulatedFeed(SimConfig(symbol="EUR/USD", bars=600, seed=21)),
            strategy=build("ma_crossover", fast=5, slow=20),
            broker=BrokerConfig(spread_pips=0.0, slippage_pips=0.0),
        )
        pricey = run_backtest(
            feed=SimulatedFeed(SimConfig(symbol="EUR/USD", bars=600, seed=21)),
            strategy=build("ma_crossover", fast=5, slow=20),
            broker=BrokerConfig(spread_pips=3.0, slippage_pips=1.0),
        )
        assert pricey.stats.net_pnl < cheap.stats.net_pnl


class TestStats:
    def test_empty_run_returns_zeroed_stats(self):
        stats = compute_stats([], [(START, 100_000.0)], 100_000.0)
        assert stats.trades == 0
        assert stats.final_equity == pytest.approx(100_000.0)

    def test_win_rate_and_profit_factor(self):
        trades = [
            Trade("EUR/USD", Side.BUY, 1000, 1.0, 1.1, START, START, 100.0),
            Trade("EUR/USD", Side.BUY, 1000, 1.0, 0.9, START, START, -50.0),
            Trade("EUR/USD", Side.BUY, 1000, 1.0, 1.2, START, START, 200.0),
        ]
        stats = compute_stats(trades, [(START, 100_000.0), (START, 100_250.0)], 100_000.0)
        assert stats.trades == 3
        assert stats.wins == 2
        assert stats.losses == 1
        assert stats.win_rate == pytest.approx(2 / 3 * 100)
        assert stats.profit_factor == pytest.approx(300 / 50)

    def test_max_drawdown_is_measured_from_the_peak(self):
        curve = [(START + timedelta(hours=i), v) for i, v in enumerate([100, 110, 90, 95, 130])]
        stats = compute_stats([], curve, 100.0)
        assert stats.max_drawdown_amount == pytest.approx(20.0)
        assert stats.max_drawdown_pct == pytest.approx(20 / 110 * 100)

    def test_expectancy_r_uses_max_r(self):
        trades = [
            Trade("EUR/USD", Side.BUY, 1000, 1.0, 1.1, START, START, 100.0, max_r=2.0),
            Trade("EUR/USD", Side.BUY, 1000, 1.0, 1.2, START, START, 200.0, max_r=4.0),
        ]
        stats = compute_stats(trades, [(START, 100_000.0), (START, 100_300.0)], 100_000.0)
        assert stats.expectancy_r == pytest.approx(3.0)

    def test_breakdowns_by_reason_and_symbol(self):
        trades = [
            Trade("EUR/USD", Side.BUY, 1000, 1.0, 1.1, START, START, 100.0, exit_reason="take_profit"),
            Trade("EUR/USD", Side.BUY, 1000, 1.0, 0.9, START, START, -50.0, exit_reason="stop_loss"),
            Trade("GBP/USD", Side.SELL, 1000, 1.0, 0.9, START, START, 100.0, exit_reason="take_profit"),
        ]
        stats = compute_stats(trades, [(START, 100_000.0)], 100_000.0)
        assert stats.by_reason["take_profit"] == (2, 200.0)
        assert stats.by_reason["stop_loss"] == (1, -50.0)
        assert stats.by_symbol["EUR/USD"] == (2, 50.0)

    def test_bars_per_year_lookup(self):
        assert BARS_PER_YEAR["1h"] == 8_760
        assert BARS_PER_YEAR["1d"] == 252


# --------------------------------------------------------------------------- #
# feeds
# --------------------------------------------------------------------------- #


class TestSimulatedFeed:
    def test_generates_the_requested_number_of_bars(self):
        feed = SimulatedFeed(SimConfig(symbol="EUR/USD", bars=100))
        assert len(list(feed)) == 100

    def test_bars_are_ordered_and_spaced(self):
        feed = SimulatedFeed(SimConfig(symbol="EUR/USD", bars=10, timeframe="1h"))
        bars = list(feed)
        for earlier, later in zip(bars, bars[1:]):
            assert later.time > earlier.time
            assert (later.time - earlier.time).total_seconds() == 3600

    def test_ohlc_is_consistent(self):
        for candle in SimulatedFeed(SimConfig(symbol="EUR/USD", bars=200)):
            assert candle.low <= min(candle.open, candle.close)
            assert candle.high >= max(candle.open, candle.close)

    def test_price_stays_positive(self):
        feed = SimulatedFeed(SimConfig(symbol="USD/JPY", start_price=150.0, bars=500))
        assert all(c.close > 0 for c in feed)

    def test_quote_has_a_spread(self):
        feed = SimulatedFeed(SimConfig(symbol="EUR/USD", bars=10))
        quote = feed.quote()
        assert quote.ask > quote.bid
        assert quote.mid == pytest.approx((quote.ask + quote.bid) / 2)


class TestCsvFeed:
    def test_round_trips_through_a_csv(self, tmp_path):
        path = generate_csv(str(tmp_path / "test.csv"), symbol="EUR/USD", bars=50, seed=7)
        feed = CsvFeed(path, symbol="EUR/USD")
        bars = list(feed)
        assert len(bars) == 50
        assert bars[0].symbol == "EUR/USD"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CsvFeed(str(tmp_path / "nope.csv"))

    def test_sorted_by_time(self, tmp_path):
        path = generate_csv(str(tmp_path / "t.csv"), symbol="EUR/USD", bars=30)
        bars = list(CsvFeed(path))
        assert bars == sorted(bars, key=lambda c: c.time)


# --------------------------------------------------------------------------- #
# storage + config
# --------------------------------------------------------------------------- #


class TestStorage:
    def test_round_trips_a_trade(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "bot.db"))
        trade = Trade(
            symbol="EUR/USD", side=Side.BUY, qty=10_000, entry_price=1.1000,
            exit_price=1.1050, entry_time=START, exit_time=START + timedelta(hours=1),
            pnl=50.0, commission=1.0, exit_reason="take_profit", tags=["a"], max_r=1.5,
        )
        store.record_trade(trade)
        stored = store.trades()
        assert len(stored) == 1
        assert stored[0].symbol == "EUR/USD"
        assert stored[0].pnl == pytest.approx(50.0)
        assert stored[0].exit_reason == "take_profit"
        assert stored[0].max_r == pytest.approx(1.5)
        store.close()

    def test_equity_curve_round_trip(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "bot.db"))
        store.record_equity(START, 100_000.0, 99_000.0, 1_000.0)
        store.record_equity(START + timedelta(hours=1), 101_000.0, 101_000.0, 0.0)
        curve = store.equity_curve()
        assert len(curve) == 2
        assert curve[-1][1] == pytest.approx(101_000.0)
        store.close()

    def test_candles_are_stored(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "bot.db"))
        store.record_candles(candle_series("EUR/USD", [1.10, 1.11, 1.12]))
        assert store.candle_count() == 3
        store.close()

    def test_events_are_stored(self, tmp_path):
        store = SQLiteStore(str(tmp_path / "bot.db"))
        store.record_event(START, "entry", {"symbol": "EUR/USD", "qty": 1000})
        events = store.events()
        assert events[0]["kind"] == "entry"
        assert events[0]["payload"]["symbol"] == "EUR/USD"
        store.close()


class TestConfig:
    def test_defaults_round_trip(self):
        config = BotConfig()
        assert BotConfig.from_dict(config.to_dict()).broker == config.broker
        assert BotConfig.from_dict(config.to_dict()).risk == config.risk

    def test_scaling_config_round_trips(self):
        config = BotConfig()
        config.scaling = ScalingConfig.pyramid(tranches=3, breakeven_at_r=1.0)
        restored = BotConfig.from_dict(config.to_dict())
        assert restored.scaling.tranches == 3
        assert restored.scaling.mode == "pyramid"
        assert restored.scaling.breakeven_at_r == 1.0

    def test_saves_and_loads_a_file(self, tmp_path):
        path = tmp_path / "config.json"
        BotConfig().save(str(path))
        assert BotConfig.load(str(path)).broker.starting_cash == pytest.approx(100_000.0)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BotConfig.load(str(tmp_path / "missing.json"))

    def test_oanda_credentials_flag(self, monkeypatch):
        monkeypatch.delenv("OANDA_TOKEN", raising=False)
        monkeypatch.delenv("OANDA_ACCOUNT", raising=False)
        assert has_oanda_credentials() is False
        monkeypatch.setenv("OANDA_TOKEN", "x")
        monkeypatch.setenv("OANDA_ACCOUNT", "y")
        assert has_oanda_credentials() is True
