"""Engine integration tests - no look-ahead, flip exits, risk gating, events."""
from __future__ import annotations

from datetime import datetime, timezone

from bot.brokers.paper import BrokerConfig, PaperBroker
from bot.core.engine import Engine, EngineConfig
from bot.core.risk import RiskConfig, RiskManager
from bot.data.feed import SimConfig, SimulatedFeed
from bot.scaling.engine import ScalingConfig, ScalingEngine
from bot.strategies import build
from tests.conftest import candle_series

START = datetime(2024, 1, 2, tzinfo=timezone.utc)


def build_engine(strategy_name="ma_crossover", scaling=None, risk=None, warmup=15, **engine_kwargs):
    broker = PaperBroker(BrokerConfig(starting_cash=100_000.0, spread_pips=1.0, slippage_pips=0.2))
    risk_manager = RiskManager(risk or RiskConfig(), 100_000.0)
    scaling_engine = ScalingEngine(scaling or ScalingConfig(), broker, risk=risk_manager)
    strategy = build(strategy_name, fast=3, slow=10)
    strategy.warmup = warmup  # keep the test series short
    engine = Engine(
        broker=broker,
        strategies=[strategy],
        risk=risk_manager,
        scaling=scaling_engine,
        config=EngineConfig(record_equity=True, **engine_kwargs),
    )
    scaling_engine.on_event = engine.emit
    return engine


def feed_series(engine, closes, symbol="EUR/USD"):
    for candle in candle_series(symbol, closes):
        engine.on_candle(candle)
    return engine


class TestWarmup:
    def test_no_trades_during_warmup(self):
        engine = build_engine(warmup=15)
        # fewer bars than the warmup -> nothing may be traded
        feed_series(engine, [1.10] * 10)
        assert engine.trades() == []
        assert not engine.broker.positions()
        # but the equity curve still records every bar
        assert len(engine.equity_curve) == 10

    def test_warmup_bars_are_still_recorded(self):
        engine = build_engine()
        feed_series(engine, [1.10] * 70)
        assert len(engine.equity_curve) == 70


class TestEntry:
    def test_enters_on_a_signal(self):
        engine = build_engine()
        feed_series(engine, [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16])
        entries = [e for e in engine.events if e["kind"] == "entry"]
        assert entries
        assert entries[0]["side"] == "BUY"
        assert entries[0]["qty"] > 0

    def test_entry_carries_a_stop(self):
        engine = build_engine()
        feed_series(engine, [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16])
        entry = next(e for e in engine.events if e["kind"] == "entry")
        assert entry["stop"] is not None
        assert entry["stop"] < entry["price"]

    def test_no_look_ahead_the_first_bar_cannot_trade(self):
        """A strategy must never see a bar before it exists."""
        engine = build_engine()
        engine.on_candle(candle_series("EUR/USD", [1.10])[0])
        assert engine.trades() == []
        assert not engine.broker.positions()

    def test_one_position_per_symbol(self):
        engine = build_engine()
        feed_series(engine, [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18])
        assert len(engine.broker.positions()) <= 1

    def test_equity_curve_grows_one_point_per_bar(self):
        engine = build_engine()
        feed_series(engine, [1.10] * 40)
        assert len(engine.equity_curve) == 40


class TestRiskGating:
    def test_blocks_when_positions_are_maxed(self):
        # max_positions=0 means every entry is refused -> the gate must fire
        engine = build_engine(risk=RiskConfig(max_positions=0, lot_size=1_000))
        feed_series(engine, [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16])
        blocks = [e for e in engine.events if e["kind"] == "risk_block"]
        assert blocks

    def test_daily_loss_limit_halts_new_entries(self):
        # an absurdly tight daily limit halts after the first losing trade
        engine = build_engine(risk=RiskConfig(max_daily_loss_pct=0.001, risk_per_trade_pct=1.0))
        feed_series(engine, [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.15, 1.14])
        assert engine.risk.halted

    def test_risk_manager_sees_every_trade(self):
        engine = build_engine()
        feed_series(engine, [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.15, 1.14])
        assert engine.risk.peak_equity >= 100_000.0


class TestExit:
    def test_signal_flip_closes_the_position(self):
        engine = build_engine()
        # rally into a long, then slide into a short
        feed_series(
            engine,
            [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16]
            + [1.15, 1.14, 1.13, 1.12, 1.11, 1.10, 1.09],
        )
        flips = [e for e in engine.events if e["kind"] == "exit_flip"]
        assert flips

    def test_exit_on_flip_can_be_disabled(self):
        engine = build_engine(exit_on_flip=False)
        feed_series(
            engine,
            [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16]
            + [1.15, 1.14, 1.13, 1.12, 1.11, 1.10, 1.09],
        )
        assert [e for e in engine.events if e["kind"] == "exit_flip"] == []

    def test_cooldown_blocks_immediate_reentry(self):
        engine = build_engine(cooldown_bars=50)
        feed_series(
            engine,
            [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16]
            + [1.15, 1.14, 1.13, 1.12, 1.11, 1.10, 1.09],
        )
        entries = [e for e in engine.events if e["kind"] == "entry"]
        assert len(entries) <= 1


class TestScalingIntegration:
    def test_scaling_events_reach_the_engine_log(self):
        engine = build_engine(scaling=ScalingConfig.pyramid(tranches=3, step_r=1.0))
        feed_series(
            engine,
            [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20],
        )
        kinds = {e["kind"] for e in engine.events}
        assert "attach" in kinds
        assert "scale_in" in kinds or "scale_out" in kinds

    def test_scaling_bot_adds_to_a_winner(self):
        engine = build_engine(scaling=ScalingConfig.pyramid(tranches=3, step_r=0.5, scale_out=()))
        feed_series(
            engine,
            [1.10] * 15 + [1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24],
        )
        adds = [e for e in engine.events if e["kind"] == "scale_in"]
        assert adds
        position = engine.broker.positions().get("EUR/USD")
        if position:
            assert position.adds >= 1

    def test_no_scaling_events_when_disabled(self):
        engine = build_engine(scaling=ScalingConfig(tranches=1))
        feed_series(engine, [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17])
        assert [e for e in engine.events if e["kind"] == "scale_in"] == []


class TestDeterminism:
    def test_same_input_same_output(self):
        closes = [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.15, 1.14, 1.13]

        def run():
            engine = build_engine()
            feed_series(engine, closes)
            return engine.equity(), [t.net_pnl for t in engine.trades()]

        assert run() == run()

    def test_simulated_feed_is_reproducible(self):
        one = list(SimulatedFeed(SimConfig(symbol="EUR/USD", bars=50, seed=42)))
        two = list(SimulatedFeed(SimConfig(symbol="EUR/USD", bars=50, seed=42)))
        assert [c.close for c in one] == [c.close for c in two]

    def test_different_seeds_differ(self):
        one = list(SimulatedFeed(SimConfig(symbol="EUR/USD", bars=50, seed=1)))
        two = list(SimulatedFeed(SimConfig(symbol="EUR/USD", bars=50, seed=2)))
        assert [c.close for c in one] != [c.close for c in two]


class TestClosePosition:
    def test_manual_close_records_a_trade(self):
        engine = build_engine()
        feed_series(engine, [1.10] * 15 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16])
        assert engine.broker.positions()
        engine.close_position("EUR/USD", "manual")
        assert not engine.broker.positions()
        assert engine.trades()[-1].exit_reason == "manual"

    def test_closing_a_flat_book_is_a_noop(self):
        engine = build_engine()
        assert engine.close_position("EUR/USD", "manual") is None
