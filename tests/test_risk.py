"""Risk manager tests - sizing, tranche splitting, exposure caps and kill switches."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.core.models import Position, Side
from bot.core.risk import RiskConfig, RiskManager, RiskViolation

START = datetime(2024, 1, 2, tzinfo=timezone.utc)


def make_risk(**kwargs) -> RiskManager:
    """Risk manager with sensible defaults; any keyword overrides them."""
    defaults = dict(
        risk_per_trade_pct=1.0,
        max_daily_loss_pct=3.0,
        max_drawdown_pct=10.0,
        max_open_exposure_pct=100.0,
        max_positions=4,
        lot_size=1_000.0,
    )
    defaults.update(kwargs)
    return RiskManager(RiskConfig(**defaults), starting_equity=100_000.0)


def position(symbol="EUR/USD", qty=100_000, price=1.10, side=Side.BUY) -> Position:
    pos = Position(symbol=symbol, side=side)
    pos.qty = qty
    pos.avg_price = price
    pos.initial_qty = qty
    return pos


class TestSizing:
    def test_usd_quoted_risk_of_one_percent(self):
        risk = make_risk()
        # 1% of 100k = 1000 at risk over a 100 pip stop -> 100,000 units
        qty = risk.position_size("EUR/USD", 100_000, 1.1000, 1.0900)
        # 1% of 100k over a 100 pip stop = 100,000 units (lot rounding + float noise)
        assert qty == pytest.approx(100_000, rel=0.02)

    def test_usd_based_pair_needs_more_units(self):
        risk = make_risk()
        # 1000 at risk over a 100 pip (1.00) stop on USD/JPY at 150 -> 150,000 units
        qty = risk.position_size("USD/JPY", 100_000, 150.00, 149.00)
        assert qty == pytest.approx(150_000, rel=0.02)

    def test_stop_out_actually_costs_the_risk_budget(self):
        from bot.core.models import spec_for

        risk = make_risk()
        entry, stop = 1.1000, 1.0900
        qty = risk.position_size("EUR/USD", 100_000, entry, stop)
        loss = -spec_for("EUR/USD").pnl(qty, entry, stop, Side.BUY)
        # lot rounding lands the realised risk within one lot (1,000 units) of budget
        assert loss == pytest.approx(1_000, rel=0.02)

    def test_tranches_split_the_risk_budget(self):
        risk = make_risk()
        single = risk.position_size("EUR/USD", 100_000, 1.1000, 1.0900, tranches=1)
        third = risk.position_size("EUR/USD", 100_000, 1.1000, 1.0900, tranches=3)
        assert third == pytest.approx(single / 3, rel=1e-2)

    def test_strength_scales_the_size(self):
        risk = make_risk()
        full = risk.position_size("EUR/USD", 100_000, 1.1000, 1.0900, strength=1.0)
        half = risk.position_size("EUR/USD", 100_000, 1.1000, 1.0900, strength=0.5)
        assert half == pytest.approx(full / 2, rel=0.02)

    def test_no_stop_means_no_position(self):
        risk = make_risk()
        assert risk.position_size("EUR/USD", 100_000, 1.1000, None) == 0.0

    def test_zero_distance_means_no_position(self):
        risk = make_risk()
        assert risk.position_size("EUR/USD", 100_000, 1.1000, 1.1000) == 0.0

    def test_rounds_down_to_lots(self):
        risk = make_risk()
        qty = risk.position_size("EUR/USD", 100_000, 1.1000, 1.0901)
        assert qty % 1_000 == 0


class TestExposure:
    def test_headroom_uses_the_remaining_budget(self):
        risk = make_risk()
        headroom = risk.max_qty_for_exposure("EUR/USD", 100_000, 0.0, 1.10)
        assert headroom == pytest.approx(90_000)

    def test_headroom_shrinks_with_open_notional(self):
        risk = make_risk()
        headroom = risk.max_qty_for_exposure("EUR/USD", 100_000, 55_000, 1.10)
        assert headroom == pytest.approx(40_000)

    def test_no_headroom_when_the_cap_is_spent(self):
        risk = make_risk()
        assert risk.max_qty_for_exposure("EUR/USD", 100_000, 100_000, 1.10) == 0.0

    def test_can_open_refuses_when_cap_is_spent(self):
        risk = make_risk()
        positions = {"EUR/USD": position(qty=100_000, price=1.10)}  # 110,000 notional
        ok, reason = risk.can_open(100_000, positions, START)
        assert not ok
        assert "exposure" in reason

    def test_can_open_refuses_at_max_positions(self):
        risk = make_risk(max_positions=2)
        positions = {
            "EUR/USD": position(qty=1_000, price=1.10),
            "GBP/USD": position("GBP/USD", 1_000, 1.26),
        }
        ok, reason = risk.can_open(100_000, positions, START)
        assert not ok
        assert "max positions" in reason

    def test_assert_raises(self):
        risk = make_risk(max_positions=0)
        with pytest.raises(RiskViolation):
            risk.assert_can_open(100_000, {})


class TestKillSwitches:
    def test_daily_loss_limit_halts(self):
        risk = make_risk()
        risk.register_equity(100_000, START)
        risk.register_equity(96_000, START + timedelta(hours=1))
        assert risk.halted
        assert "daily loss" in risk.halt_reason

    def test_drawdown_limit_halts(self):
        # raise the daily cap so only the drawdown switch can fire
        risk = make_risk(max_drawdown_pct=10.0, max_daily_loss_pct=50.0)
        risk.register_equity(120_000, START)          # new high water mark
        risk.register_equity(107_000, START + timedelta(hours=1))
        assert risk.halted
        assert "drawdown" in risk.halt_reason

    def test_small_drawdown_does_not_halt(self):
        risk = make_risk(max_daily_loss_pct=50.0)
        risk.register_equity(120_000, START)
        risk.register_equity(115_000, START + timedelta(hours=1))
        assert not risk.halted

    def test_halted_manager_blocks_new_trades(self):
        risk = make_risk()
        risk.register_equity(100_000, START)
        risk.register_equity(90_000, START + timedelta(hours=1))
        ok, reason = risk.can_open(90_000, {}, START + timedelta(hours=2))
        assert not ok
        assert "halted" in reason

    def test_unhalt_clears_the_flag(self):
        risk = make_risk()
        risk._halt("test")
        risk.unhalt()
        assert not risk.halted

    def test_new_day_resets_the_daily_counter(self):
        risk = make_risk()
        risk.register_equity(100_000, START)
        risk.register_equity(96_500, START + timedelta(hours=1))
        # next calendar day, equity back up -> the daily loss no longer applies
        risk.reset_daily(96_500)
        risk.unhalt()  # the daily halt is sticky until cleared
        ok, _ = risk.can_open(96_500, {}, START + timedelta(days=1))
        assert ok

    def test_drawdown_reporting(self):
        risk = make_risk()
        risk.register_equity(100_000, START)
        risk.register_equity(110_000, START + timedelta(hours=1))
        assert risk.drawdown_pct(104_500) == pytest.approx(5.0)
        assert risk.peak_equity == pytest.approx(110_000)
