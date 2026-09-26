"""Scaling bot tests - scale-in timing, scale-out ladder, breakeven and risk control.

Everything is driven through a :class:`~bot.brokers.paper.PaperBroker` with a
scripted price path, so the assertions are about behaviour, not arithmetic.
"""
from __future__ import annotations

import pytest

from bot.brokers.paper import BrokerConfig, PaperBroker
from bot.core.models import Order, OrderType, Side, spec_for
from bot.scaling.engine import ManagedPosition, ScaleOutLeg, ScalingConfig, ScalingEngine
from tests.conftest import candle_series, make_candle


def make_broker(**kwargs) -> PaperBroker:
    return PaperBroker(BrokerConfig(starting_cash=100_000.0, spread_pips=1.0, slippage_pips=0.2, **kwargs))


STOP = 1.09000


def open_long(broker, price=1.10000, qty=30_000, stop=STOP, index=0):
    """Feed a bar to set the price, then open a long with a stop.

    Returns the fill: the real entry is ask + slippage, so ``R`` is a little
    larger than ``price - stop`` and tests must measure R from the fill.
    """
    broker.on_candle(make_candle("EUR/USD", price, index=index))
    fill = broker.submit(
        Order("EUR/USD", Side.BUY, qty, order_type=OrderType.MARKET, stop_loss=stop, tag="entry")
    )
    assert fill is not None
    return fill


def r_price(fill, multiple=1.0, stop=STOP):
    """Price exactly ``multiple`` R away from the real entry price."""
    r = abs(fill.price - stop)
    return fill.price + multiple * r


@pytest.fixture
def broker():
    return make_broker()


class TestConfig:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            ScalingConfig(mode="sideways")

    def test_rejects_scale_out_over_one_hundred_percent(self):
        with pytest.raises(ValueError):
            ScalingConfig(scale_out=(ScaleOutLeg(1.0, 0.6), ScaleOutLeg(2.0, 0.6)))

    def test_sorts_the_scale_out_ladder(self):
        cfg = ScalingConfig(scale_out=(ScaleOutLeg(3.0, 0.25), ScaleOutLeg(1.0, 0.5)))
        assert [leg.r for leg in cfg.scale_out] == [1.0, 3.0]

    def test_round_trips_through_dict(self):
        cfg = ScalingConfig.pyramid(tranches=3, step_r=1.5, breakeven_at_r=1.0, trail_atr_mult=3.0)
        assert ScalingConfig.from_dict(cfg.to_dict()) == cfg

    def test_pyramid_and_dca_factories(self):
        assert ScalingConfig.pyramid().mode == "pyramid"
        assert ScalingConfig.dca().mode == "dca"
        assert ScalingConfig.dca().add_step_r == 0.5

    def test_off_by_default(self):
        cfg = ScalingConfig()
        assert cfg.tranches == 1
        assert cfg.scale_out == ()


class TestProgress:
    def test_progress_is_measured_in_r(self):
        mp = ManagedPosition(
            symbol="EUR/USD", side=Side.BUY, first_entry=1.10, initial_stop=1.09, initial_qty=10_000
        )
        assert mp.r == pytest.approx(0.01)
        assert mp.progress(1.11) == pytest.approx(1.0)
        assert mp.progress(1.12) == pytest.approx(2.0)
        assert mp.progress(1.09) == pytest.approx(-1.0)

    def test_progress_for_shorts_is_inverted(self):
        mp = ManagedPosition(
            symbol="EUR/USD", side=Side.SELL, first_entry=1.10, initial_stop=1.11, initial_qty=10_000
        )
        assert mp.progress(1.09) == pytest.approx(1.0)   # price fell -> good for a short
        assert mp.progress(1.11) == pytest.approx(-1.0)

    def test_extremes_track_best_and_worst_r(self):
        mp = ManagedPosition(
            symbol="EUR/USD", side=Side.BUY, first_entry=1.10, initial_stop=1.09, initial_qty=10_000
        )
        for price in (1.105, 1.095, 1.115, 1.09):
            mp.update_extremes(price)
        assert mp.max_progress_r == pytest.approx(1.5)
        assert mp.min_progress_r == pytest.approx(-1.0)
        assert mp.peak == pytest.approx(1.115)
        assert mp.trough == pytest.approx(1.09)


class TestScaleIn:
    def test_pyramid_adds_after_one_r_in_favour(self, broker):
        fill = open_long(broker, qty=30_000)
        scaling = ScalingEngine(
            ScalingConfig.pyramid(tranches=3, step_r=1.0, scale_out=()), broker
        )
        # one full R in our favour is the first add
        orders = scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))
        assert len(orders) == 1
        assert orders[0].side is Side.BUY
        assert orders[0].tag == "scalein#2"

    def test_pyramid_does_not_add_before_the_step(self, broker):
        fill = open_long(broker, qty=30_000)
        scaling = ScalingEngine(ScalingConfig.pyramid(tranches=3, step_r=1.0), broker)
        assert scaling.on_candle(make_candle("EUR/USD", r_price(fill, 0.9), index=1)) == []

    def test_dca_adds_when_price_moves_against(self, broker):
        fill = open_long(broker, qty=30_000)
        # scale_out disabled so this test isolates the scale-in decision
        scaling = ScalingEngine(ScalingConfig.dca(tranches=3, step_r=0.5, scale_out=()), broker)
        # half an R against us -> add
        orders = scaling.on_candle(make_candle("EUR/USD", r_price(fill, -0.6), index=1))
        assert len(orders) == 1
        assert orders[0].tag == "scalein#2"

    def test_dca_does_not_add_when_price_rallies(self, broker):
        fill = open_long(broker, qty=30_000)
        scaling = ScalingEngine(ScalingConfig.dca(tranches=3, step_r=0.5, scale_out=()), broker)
        assert scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.5), index=1)) == []

    def test_stops_adding_after_the_last_tranche(self, broker):
        fill = open_long(broker, qty=30_000)
        scaling = ScalingEngine(ScalingConfig.pyramid(tranches=2, step_r=1.0, scale_out=()), broker)
        assert len(scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))) == 1
        # tranches=2 means one add only, however far price runs
        assert scaling.on_candle(make_candle("EUR/USD", r_price(fill, 3.0), index=2)) == []

    def test_single_tranche_config_never_adds(self, broker):
        fill = open_long(broker, qty=30_000)
        scaling = ScalingEngine(ScalingConfig(tranches=1), broker)
        assert scaling.on_candle(make_candle("EUR/USD", r_price(fill, 2.0), index=1)) == []

    def test_adds_are_skipped_when_the_exposure_cap_is_spent(self):
        broker = make_broker()
        # a position that already uses the whole exposure budget
        fill = open_long(broker, price=1.10000, qty=90_909, stop=1.09000)
        scaling = ScalingEngine(ScalingConfig.pyramid(tranches=3, step_r=1.0), broker)
        scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0, stop=1.09000), index=1))
        assert scaling.stats["skipped_adds"] >= 1
        assert scaling.stats["adds"] == 0


class TestScaleOut:
    def test_closes_the_configured_fraction_at_one_r(self, broker):
        fill = open_long(broker, qty=100_000)
        cfg = ScalingConfig(scale_out=(ScaleOutLeg(1.0, 0.5), ScaleOutLeg(2.0, 0.5)))
        scaling = ScalingEngine(cfg, broker)
        orders = scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))
        assert len(orders) == 1
        assert orders[0].reduce_only is True
        assert orders[0].side is Side.SELL
        assert orders[0].qty == pytest.approx(50_000)

    def test_each_leg_fires_only_once(self, broker):
        fill = open_long(broker, qty=100_000)
        cfg = ScalingConfig(scale_out=(ScaleOutLeg(1.0, 0.5), ScaleOutLeg(2.0, 0.5)))
        scaling = ScalingEngine(cfg, broker)
        scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))   # 1R leg
        scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.2), index=2))   # no repeat
        orders = scaling.on_candle(make_candle("EUR/USD", r_price(fill, 2.0), index=3))
        assert len(orders) == 1
        assert orders[0].qty == pytest.approx(50_000)

    def test_ladder_empties_the_position(self, broker):
        fill = open_long(broker, qty=100_000)
        cfg = ScalingConfig(scale_out=(ScaleOutLeg(1.0, 0.5), ScaleOutLeg(2.0, 0.5)))
        scaling = ScalingEngine(cfg, broker)
        for order in scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1)):
            broker.submit(order)
        for order in scaling.on_candle(make_candle("EUR/USD", r_price(fill, 2.0), index=2)):
            broker.submit(order)
        assert broker.positions() == {}

    def test_ladder_closes_exactly_the_position(self, broker):
        fill = open_long(broker, qty=20_000)
        cfg = ScalingConfig(scale_out=(ScaleOutLeg(1.0, 0.6), ScaleOutLeg(2.0, 0.4)))
        scaling = ScalingEngine(cfg, broker)
        for order in scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1)):
            broker.submit(order)
        for order in scaling.on_candle(make_candle("EUR/USD", r_price(fill, 2.0), index=2)):
            broker.submit(order)
        assert broker.positions() == {}
        assert sum(t.qty for t in broker.closed_trades()) == pytest.approx(20_000)

    def test_ladder_rejects_fractions_over_one_hundred_percent(self):
        with pytest.raises(ValueError):
            ScalingConfig(scale_out=(ScaleOutLeg(1.0, 0.75), ScaleOutLeg(2.0, 0.75)))


class TestBreakevenAndTrail:
    def test_breakeven_moves_the_stop_to_entry(self, broker):
        fill = open_long(broker, qty=30_000, stop=1.09000)
        scaling = ScalingEngine(ScalingConfig(breakeven_at_r=1.0), broker)
        scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))
        assert broker.positions()["EUR/USD"].stop_loss == pytest.approx(fill.price)

    def test_breakeven_fires_only_once(self, broker):
        fill = open_long(broker, qty=30_000, stop=1.09000)
        scaling = ScalingEngine(ScalingConfig(breakeven_at_r=1.0), broker)
        scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))
        scaling.on_candle(make_candle("EUR/USD", r_price(fill, 2.0), index=2))
        assert scaling.stats["breakevens"] == 1

    def test_breakeven_not_triggered_below_the_threshold(self, broker):
        fill = open_long(broker, qty=30_000, stop=1.09000)
        scaling = ScalingEngine(ScalingConfig(breakeven_at_r=1.0), broker)
        scaling.on_candle(make_candle("EUR/USD", r_price(fill, 0.9), index=1))
        assert broker.positions()["EUR/USD"].stop_loss == pytest.approx(1.09000)

    def test_trailing_stop_ratchets_up_for_a_long(self, broker):
        fill = open_long(broker, qty=30_000, stop=1.09000)
        cfg = ScalingConfig(trail_atr_mult=1.0, trail_start_r=1.0)
        scaling = ScalingEngine(cfg, broker, atr_period=3)
        series = candle_series(
            "EUR/USD", [r_price(fill, 0.5), r_price(fill, 1.2), r_price(fill, 1.6), r_price(fill, 2.0)],
            start_index=1,
        )
        stops = []
        for i, candle in enumerate(series):
            scaling.on_candle(candle, series[: i + 1])
            stops.append(broker.positions()["EUR/USD"].stop_loss)
        assert stops[-1] > 1.09000
        assert scaling.stats["trail_updates"] > 0

    def test_time_stop_closes_the_position(self, broker):
        open_long(broker, qty=30_000)
        cfg = ScalingConfig(time_stop_bars=3)
        scaling = ScalingEngine(cfg, broker)
        closed = False
        for i in range(1, 6):
            for order in scaling.on_candle(make_candle("EUR/USD", 1.10100, index=i)):
                broker.submit(order)
                closed = True
        assert closed
        assert broker.positions() == {}
        assert broker.closed_trades()[-1].exit_reason == "time_stop"


class TestConstantRisk:
    def test_add_keeps_total_open_risk_constant(self, broker):
        """After a scale-in the stop is re-solved so the whole position still
        risks roughly what the first tranche risked."""
        entry = 1.10000
        stop = 1.09000
        fill = open_long(broker, price=entry, qty=30_000, stop=stop)
        spec = spec_for("EUR/USD")
        initial_risk = -spec.pnl(30_000, fill.price, stop, Side.BUY)

        cfg = ScalingConfig.pyramid(tranches=2, step_r=1.0, add_size_mult=1.0, scale_out=())
        scaling = ScalingEngine(cfg, broker)
        orders = scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))
        assert len(orders) == 1
        broker.submit(orders[0])
        scaling.on_scale_in_filled("EUR/USD", orders[0].qty and r_price(fill, 1.0), orders[0].qty)

        position = broker.positions()["EUR/USD"]
        assert position.qty == pytest.approx(60_000)
        new_risk = -spec.pnl(position.qty, position.avg_price, position.stop_loss, Side.BUY)
        # risk is held near the original budget rather than doubling with the add
        assert new_risk == pytest.approx(initial_risk, rel=0.15)

    def test_stop_never_moves_through_the_entry(self, broker):
        fill = open_long(broker, price=1.10000, qty=30_000, stop=1.09000)
        cfg = ScalingConfig.pyramid(tranches=2, step_r=1.0)
        scaling = ScalingEngine(cfg, broker)
        orders = scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))
        broker.submit(orders[0])
        scaling.on_scale_in_filled("EUR/USD", r_price(fill, 1.0), orders[0].qty)
        assert broker.positions()["EUR/USD"].stop_loss <= fill.price


class TestLifecycle:
    def test_attaches_to_an_existing_position(self, broker):
        open_long(broker, qty=30_000)
        scaling = ScalingEngine(ScalingConfig(), broker)
        events = []
        scaling.on_event = lambda kind, **payload: events.append(kind)
        scaling.on_candle(make_candle("EUR/USD", 1.10100, index=1))
        assert scaling.is_managing("EUR/USD")
        assert "attach" in events

    def test_releases_when_the_position_closes(self, broker):
        open_long(broker, qty=30_000, stop=1.09000)
        scaling = ScalingEngine(ScalingConfig(), broker)
        scaling.on_candle(make_candle("EUR/USD", 1.10100, index=1))
        assert scaling.is_managing("EUR/USD")
        broker.on_candle(make_candle("EUR/USD", 1.0950, index=2, low=1.0890))  # stopped out
        scaling.on_candle(make_candle("EUR/USD", 1.0950, index=3))
        assert not scaling.is_managing("EUR/USD")

    def test_no_position_means_no_orders(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        scaling = ScalingEngine(ScalingConfig.pyramid(), broker)
        assert scaling.on_candle(make_candle("EUR/USD", 1.11000, index=1)) == []
        assert scaling.managed == {}

    def test_events_carry_the_payload(self, broker):
        fill = open_long(broker, qty=100_000)
        scaling = ScalingEngine(ScalingConfig(scale_out=(ScaleOutLeg(1.0, 0.5),)), broker)
        seen = []
        scaling.on_event = lambda kind, **payload: seen.append((kind, payload))
        scaling.on_candle(make_candle("EUR/USD", r_price(fill, 1.0), index=1))
        kinds = [kind for kind, _ in seen]
        assert "scale_out" in kinds
        payload = dict(seen[kinds.index("scale_out")][1])
        assert payload["symbol"] == "EUR/USD"
        assert payload["qty"] == pytest.approx(50_000)
