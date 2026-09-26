"""Domain model tests."""
from __future__ import annotations

import pytest

from bot.core.models import (
    Candle,
    Fill,
    Order,
    OrderType,
    Position,
    Side,
    spec_for,
    stop_for_risk,
    round_to_lot,
    utcnow,
)
from tests.conftest import make_candle


class TestSide:
    def test_signs(self):
        assert Side.BUY.sign == 1
        assert Side.SELL.sign == -1

    def test_opposite(self):
        assert Side.BUY.opposite is Side.SELL
        assert Side.SELL.opposite is Side.BUY


class TestSymbolSpec:
    def test_detects_jpy_pip_size(self):
        assert spec_for("USD/JPY").pip_size == 0.01
        assert spec_for("EUR/USD").pip_size == 0.0001
        assert spec_for("gbpjpy").pip_size == 0.01

    def test_normalises_symbol_format(self):
        spec = spec_for("eur_usd")
        assert spec.symbol == "EUR/USD"
        assert spec.base_ccy == "EUR"
        assert spec.quote_ccy == "USD"

    def test_concatenated_symbol(self):
        spec = spec_for("EURUSD")
        assert spec.base_ccy == "EUR"
        assert spec.quote_ccy == "USD"

    def test_pips_helper(self):
        spec = spec_for("EUR/USD")
        assert spec.pips(0.0030) == pytest.approx(30.0)
        assert spec_for("USD/JPY").pips(1.0) == pytest.approx(100.0)


class TestPnl:
    def test_usd_quoted_long(self):
        spec = spec_for("EUR/USD")
        # 100k EUR bought at 1.1000, sold at 1.1050 -> +500 USD
        assert spec.pnl(100_000, 1.1000, 1.1050, Side.BUY) == pytest.approx(500.0)

    def test_usd_quoted_short_profits_when_price_falls(self):
        spec = spec_for("EUR/USD")
        # shorted at 1.1050, price fell to 1.1000 -> +500 USD
        assert spec.pnl(100_000, 1.1050, 1.1000, Side.SELL) == pytest.approx(500.0)

    def test_usd_based_long(self):
        spec = spec_for("USD/JPY")
        # 100k USD bought at 150.00, sold at 151.00 -> +100,000 JPY -> /151 = USD
        expected = 100_000 * 1.0 / 151.0
        assert spec.pnl(100_000, 150.00, 151.00, Side.BUY) == pytest.approx(expected)

    def test_usd_based_short_profits_when_price_falls(self):
        spec = spec_for("USD/JPY")
        # shorted at 151.00, price fell to 150.00 -> +100,000 JPY -> /150 exit rate
        expected = 100_000 * 1.0 / 150.0
        assert spec.pnl(100_000, 151.00, 150.00, Side.SELL) == pytest.approx(expected)


class TestStopForRisk:
    def test_usd_quoted_round_trip(self):
        spec = spec_for("EUR/USD")
        stop = stop_for_risk(spec, Side.BUY, 1.1000, 100_000, 1_000.0)
        assert stop == pytest.approx(1.0900, abs=1e-9)
        # the stop must actually cost the target amount
        assert -spec.pnl(100_000, 1.1000, stop, Side.BUY) == pytest.approx(1_000.0, rel=1e-6)

    def test_usd_based_round_trip(self):
        spec = spec_for("USD/JPY")
        stop = stop_for_risk(spec, Side.BUY, 150.00, 100_000, 1_000.0)
        assert -spec.pnl(100_000, 150.00, stop, Side.BUY) == pytest.approx(1_000.0, rel=1e-6)

    def test_short_stop_sits_above(self):
        spec = spec_for("EUR/USD")
        stop = stop_for_risk(spec, Side.SELL, 1.1000, 100_000, 1_000.0)
        assert stop > 1.1000

    def test_zero_risk_returns_entry(self):
        spec = spec_for("EUR/USD")
        assert stop_for_risk(spec, Side.BUY, 1.1000, 100_000, 0.0) == 1.1000


class TestRoundToLot:
    def test_rounds_down(self):
        assert round_to_lot(33_333, 1_000) == 33_000

    def test_never_rounds_to_zero(self):
        assert round_to_lot(500, 1_000) == 1_000

    def test_zero_stays_zero(self):
        assert round_to_lot(0) == 0.0
        assert round_to_lot(-5) == 0.0


class TestPosition:
    def _fill(self, price, qty, side=Side.BUY):
        return Fill(
            order_id=1,
            symbol="EUR/USD",
            side=side,
            qty=qty,
            price=price,
            time=utcnow(),
        )

    def test_weighted_average(self):
        position = Position(symbol="EUR/USD", side=Side.BUY)
        position.apply_fill(self._fill(1.1000, 100_000))
        position.apply_fill(self._fill(1.0900, 100_000))
        assert position.avg_price == pytest.approx(1.0950)
        assert position.qty == pytest.approx(200_000)
        assert position.adds == 1

    def test_initial_qty_remembers_first_tranche(self):
        position = Position(symbol="EUR/USD", side=Side.BUY)
        position.apply_fill(self._fill(1.1000, 50_000))
        position.apply_fill(self._fill(1.0900, 50_000))
        assert position.initial_qty == 50_000
        assert len(position.tranches) == 2

    def test_opposite_fill_rejected(self):
        position = Position(symbol="EUR/USD", side=Side.BUY)
        with pytest.raises(ValueError):
            position.apply_fill(self._fill(1.1000, 1_000, side=Side.SELL))

    def test_unrealized_pnl(self):
        position = Position(symbol="EUR/USD", side=Side.BUY)
        position.apply_fill(self._fill(1.1000, 100_000))
        assert position.unrealized(1.1050, spec_for("EUR/USD")) == pytest.approx(500.0)


class TestCandle:
    def test_rejects_inconsistent_ohlc(self):
        with pytest.raises(ValueError):
            Candle("EUR/USD", utcnow(), open=1.0, high=1.1, low=1.2, close=1.05)

    def test_accepts_valid_ohlc(self):
        candle = make_candle("EUR/USD", 1.1000)
        assert candle.high >= max(candle.open, candle.close)
        assert candle.low <= min(candle.open, candle.close)


class TestOrder:
    def test_market_order_needs_qty(self):
        with pytest.raises(ValueError):
            Order(symbol="EUR/USD", side=Side.BUY, qty=0)

    def test_limit_order_needs_price(self):
        with pytest.raises(ValueError):
            Order(symbol="EUR/USD", side=Side.BUY, qty=1000, order_type=OrderType.LIMIT)

    def test_stop_order_needs_price(self):
        with pytest.raises(ValueError):
            Order(symbol="EUR/USD", side=Side.BUY, qty=1000, order_type=OrderType.STOP)
