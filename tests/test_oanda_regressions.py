"""Regression tests for bugs found while building the system, plus the OANDA
adapter driven by a mocked transport (no network, no credentials).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.brokers.paper import BrokerConfig, PaperBroker
from bot.core.models import Side
from bot.data.oanda import GRANULARITY, OandaBroker, OandaClient, OandaError, OandaFeed
from bot.scaling.ladder import GridLadder, LadderConfig
from tests.conftest import make_candle


# --------------------------------------------------------------------------- #
# regression: the hard stop must not resurrect the ladder
# --------------------------------------------------------------------------- #


class TestHardStopRegression:
    def make_ladder(self, broker, **kwargs):
        params = dict(centre_price=1.0800, levels=3, spacing_pips=25.0,
                      qty_per_level=10_000, take_profit_pips=20.0,
                      direction="buy", stop_loss_pips=50.0)
        params.update(kwargs)
        return GridLadder(LadderConfig(**params), broker, symbol="EUR/USD")

    def test_no_spurious_fills_after_the_hard_stop(self):
        """A cancelled entry order must not be counted as a fresh fill."""
        broker = PaperBroker(BrokerConfig(starting_cash=100_000.0))
        ladder = self.make_ladder(broker)
        ladder.build()

        # price runs far past the 50 pip hard stop and stays there
        for i in range(1, 30):
            candle = make_candle("EUR/USD", 1.0600, index=i, low=1.0590, high=1.0610)
            broker.on_candle(candle)
            ladder.on_candle(candle)

        assert ladder.stopped
        fills_after_stop = ladder.stats["fills"]

        # ...and keeps running: nothing new may be counted or traded
        for i in range(30, 60):
            candle = make_candle("EUR/USD", 1.0600, index=i, low=1.0590, high=1.0610)
            broker.on_candle(candle)
            ladder.on_candle(candle)

        assert ladder.stats["fills"] == fills_after_stop
        assert not broker.positions()
        assert broker.working_orders() == []

    def test_hard_stop_flattens_an_open_position(self):
        broker = PaperBroker(BrokerConfig(starting_cash=100_000.0))
        ladder = self.make_ladder(broker)
        ladder.build()
        # fill the first rung, then crash through the hard stop
        for i, close in enumerate([1.0770, 1.0600], start=1):
            candle = make_candle("EUR/USD", close, index=i, low=close - 0.001, high=close + 0.001)
            broker.on_candle(candle)
            ladder.on_candle(candle)
        assert ladder.stopped
        assert not broker.positions()

    def test_summary_reports_the_stopped_flag(self):
        broker = PaperBroker(BrokerConfig(starting_cash=100_000.0))
        ladder = self.make_ladder(broker)
        ladder.build()
        candle = make_candle("EUR/USD", 1.0600, index=1, low=1.0590, high=1.0610)
        broker.on_candle(candle)
        ladder.on_candle(candle)
        assert ladder.summary()["stopped"] is True

    def test_ladder_without_a_hard_stop_keeps_cycling(self):
        broker = PaperBroker(BrokerConfig(starting_cash=100_000.0))
        ladder = self.make_ladder(broker, stop_loss_pips=0.0)
        ladder.build()
        # oscillate around the rungs so fills and targets both happen
        for i in range(1, 40):
            close = 1.0770 if i % 2 else 1.0800
            candle = make_candle("EUR/USD", close, index=i, low=close - 0.0005, high=close + 0.0005)
            broker.on_candle(candle)
            ladder.on_candle(candle)
        assert not ladder.stopped
        assert ladder.stats["fills"] > 0


# --------------------------------------------------------------------------- #
# regression: float noise must not hide a moving-average crossover
# --------------------------------------------------------------------------- #


class TestCrossoverRegression:
    def test_flat_then_rising_still_crosses(self):
        """Two MAs over a flat stretch can differ by ULPs; the cross must fire."""
        from bot.strategies import build
        from tests.conftest import candle_series

        strategy = build("ma_crossover", fast=3, slow=10)
        closes = [1.10] * 12 + [1.11, 1.12, 1.13, 1.14, 1.15, 1.16]
        series = candle_series("EUR/USD", closes)
        signals = [s for s in (strategy.on_candle(series[: i + 1]) for i in range(len(series))) if s]
        assert signals
        assert signals[0].side is Side.BUY


# --------------------------------------------------------------------------- #
# OANDA adapter (mocked transport)
# --------------------------------------------------------------------------- #


def fake_transport(responses):
    """Return a transport that replays canned responses and records requests."""
    calls = []

    def transport(method, url, body, headers):
        calls.append((method, url, body))
        key = f"{method} {url.split('/v3')[-1].split('?')[0]}"
        for pattern, payload in responses.items():
            if pattern in key:
                return payload
        return {}

    transport.calls = calls
    return transport


class TestOandaClient:
    def test_requires_credentials(self, monkeypatch):
        monkeypatch.delenv("OANDA_TOKEN", raising=False)
        monkeypatch.delenv("OANDA_ACCOUNT", raising=False)
        with pytest.raises(OandaError):
            OandaClient(token="", account="")

    def test_price_parses_bids_and_asks(self):
        transport = fake_transport({
            "GET /accounts/123/pricing": {
                "prices": [{"bids": [{"price": "1.09950"}], "asks": [{"price": "1.09970"}]}]
            }
        })
        client = OandaClient(token="t", account="123", transport=transport)
        quote = client.price("EUR_USD")
        assert quote.bid == pytest.approx(1.09950)
        assert quote.ask == pytest.approx(1.09970)
        assert quote.symbol == "EUR/USD"

    def test_price_with_no_data_raises(self):
        client = OandaClient(token="t", account="123", transport=fake_transport({}))
        with pytest.raises(OandaError):
            client.price("EUR_USD")

    def test_candles_are_parsed_and_skipped_when_incomplete(self):
        transport = fake_transport({
            "GET /instruments/EUR_USD/candles": {
                "candles": [
                    {"complete": True, "time": "2024-01-02T00:00:00Z", "volume": "100",
                     "mid": {"o": "1.1", "h": "1.11", "l": "1.09", "c": "1.105"}},
                    {"complete": False, "time": "2024-01-02T01:00:00Z", "volume": "50",
                     "mid": {"o": "1.1", "h": "1.12", "l": "1.09", "c": "1.115"}},
                ]
            }
        })
        client = OandaClient(token="t", account="123", transport=transport)
        bars = client.candles("EUR_USD", count=2, granularity="H1")
        assert len(bars) == 1
        assert bars[0].symbol == "EUR/USD"
        assert bars[0].close == pytest.approx(1.105)
        assert bars[0].time == datetime(2024, 1, 2, tzinfo=timezone.utc)

    def test_market_order_attaches_a_bracket(self):
        transport = fake_transport({
            "POST /accounts/123/orders": {
                "orderFillTransaction": {"price": "1.10000", "units": "10000"}
            }
        })
        client = OandaClient(token="t", account="123", transport=transport)
        client.market_order("EUR_USD", 10_000, stop_loss=1.09, take_profit=1.12)
        method, url, body = transport.calls[-1]
        assert method == "POST"
        order = body["order"]
        assert order["units"] == "10000"
        assert order["stopLossOnFill"]["price"] == "1.09000"
        assert order["takeProfitOnFill"]["price"] == "1.12000"

    def test_granularity_mapping_is_correct(self):
        assert GRANULARITY["1h"] == "H1"
        assert GRANULARITY["1d"] == "D"
        assert GRANULARITY["15m"] == "M15"


class TestOandaBroker:
    def make_broker(self, transport):
        return OandaBroker(OandaClient(token="t", account="123", transport=transport))

    def test_positions_are_mapped(self):
        transport = fake_transport({
            "GET /accounts/123/openPositions": {
                "positions": [{
                    "instrument": "EUR_USD",
                    "long": {"units": "50000", "averagePrice": "1.10000"},
                    "short": {"units": "0", "averagePrice": "1.10000"},
                }]
            }
        })
        positions = self.make_broker(transport).positions()
        assert positions["EUR/USD"].side is Side.BUY
        assert positions["EUR/USD"].qty == pytest.approx(50_000)
        assert positions["EUR/USD"].avg_price == pytest.approx(1.10)

    def test_short_positions_are_mapped(self):
        transport = fake_transport({
            "GET /accounts/123/openPositions": {
                "positions": [{
                    "instrument": "USD_JPY",
                    "long": {"units": "0", "averagePrice": "150.0"},
                    "short": {"units": "30000", "averagePrice": "150.5"},
                }]
            }
        })
        positions = self.make_broker(transport).positions()
        assert positions["USD/JPY"].side is Side.SELL
        assert positions["USD/JPY"].qty == pytest.approx(30_000)

    def test_submit_returns_a_fill(self):
        transport = fake_transport({
            "POST /accounts/123/orders": {
                "orderFillTransaction": {"price": "1.10005", "units": "10000"}
            }
        })
        from bot.core.models import Order

        fill = self.make_broker(transport).submit(Order("EUR/USD", Side.BUY, 10_000))
        assert fill is not None
        assert fill.price == pytest.approx(1.10005)
        assert fill.qty == pytest.approx(10_000)

    def test_submit_returns_none_when_the_order_is_rejected(self):
        transport = fake_transport({})
        from bot.core.models import Order

        assert self.make_broker(transport).submit(Order("EUR/USD", Side.BUY, 10_000)) is None


class TestOandaFeed:
    def test_feed_yields_candles(self):
        transport = fake_transport({
            "GET /instruments/EUR_USD/candles": {
                "candles": [
                    {"complete": True, "time": "2024-01-02T00:00:00Z", "volume": "1",
                     "mid": {"o": "1.1", "h": "1.11", "l": "1.09", "c": "1.105"}}
                ]
            }
        })
        feed = OandaFeed(OandaClient(token="t", account="123", transport=transport), "EUR_USD")
        bars = list(feed)
        assert len(bars) == 1
        assert feed.symbol == "EUR/USD"
        assert feed.timeframe == "1h"
