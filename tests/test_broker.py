"""Paper broker tests - fills, spread, stops, netting and FIFO accounting."""
from __future__ import annotations

import pytest

from bot.brokers.paper import BrokerConfig, PaperBroker
from bot.core.models import Order, OrderType, Side
from tests.conftest import make_candle


def make_broker(**kwargs) -> PaperBroker:
    config = BrokerConfig(
        starting_cash=100_000.0,
        spread_pips=1.0,
        slippage_pips=0.2,
        **kwargs,
    )
    return PaperBroker(config)


@pytest.fixture
def broker():
    return make_broker()


class TestQuoting:
    def test_buy_crosses_the_offer(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        # half spread = 0.5 pip -> bid 1.09995 / ask 1.10005, plus 0.2 pip slippage
        fill = broker.submit(Order("EUR/USD", Side.BUY, 100_000))
        assert fill.price == pytest.approx(1.10000 + 0.5 * 0.0001 + 0.2 * 0.0001)

    def test_sell_crosses_the_bid(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        fill = broker.submit(Order("EUR/USD", Side.SELL, 100_000))
        assert fill.price == pytest.approx(1.10000 - 0.5 * 0.0001 - 0.2 * 0.0001)

    def test_jpy_pair_pip_size(self):
        jpy_broker = make_broker()
        jpy_broker.on_candle(make_candle("USD/JPY", 150.000))
        fill = jpy_broker.submit(Order("USD/JPY", Side.BUY, 100_000))
        # JPY pip = 0.01, so half spread is 0.005
        assert fill.price == pytest.approx(150.000 + 0.005 + 0.002)

    def test_quote_before_any_price_raises(self, broker):
        with pytest.raises(KeyError):
            broker.quote("EUR/USD")


class TestStopsAndTargets:
    def test_long_stop_loss_fills_at_the_stop(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000, stop_loss=1.0900))
        # a later bar trades down through the stop
        broker.on_candle(make_candle("EUR/USD", 1.0950, index=1, low=1.0890, high=1.0960))
        assert broker.positions() == {}
        trades = broker.closed_trades()
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop_loss"
        assert trades[0].exit_price == pytest.approx(1.0900)

    def test_gap_through_stop_fills_at_the_open(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000, stop_loss=1.0900))
        # opens below the stop -> worse fill than the stop price
        broker.on_candle(make_candle("EUR/USD", 1.0850, index=1, low=1.0800, high=1.0900))
        trades = broker.closed_trades()
        assert trades[0].exit_price == pytest.approx(1.0850)

    def test_take_profit_fills_at_the_target(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000, take_profit=1.1200))
        broker.on_candle(make_candle("EUR/USD", 1.1150, index=1, low=1.1100, high=1.1250))
        trades = broker.closed_trades()
        assert trades[0].exit_reason == "take_profit"
        assert trades[0].exit_price == pytest.approx(1.1200)

    def test_stop_wins_when_both_are_inside_one_bar(self, broker):
        """Pessimistic assumption: if the bar spans both levels, the stop went first."""
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000, stop_loss=1.0900, take_profit=1.1200))
        broker.on_candle(make_candle("EUR/USD", 1.1050, index=1, low=1.0850, high=1.1250))
        trades = broker.closed_trades()
        assert trades[0].exit_reason == "stop_loss"

    def test_short_stop_is_above_entry(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.SELL, 100_000, stop_loss=1.1100))
        broker.on_candle(make_candle("EUR/USD", 1.1050, index=1, low=1.1040, high=1.1150))
        trades = broker.closed_trades()
        assert trades[0].exit_reason == "stop_loss"
        assert trades[0].exit_price == pytest.approx(1.1100)


class TestNetting:
    def test_opposite_order_closes_the_position(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000))
        broker.submit(Order("EUR/USD", Side.SELL, 100_000))
        assert broker.positions() == {}
        assert len(broker.closed_trades()) == 1

    def test_partial_close_leaves_the_rest(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000))
        broker.submit(Order("EUR/USD", Side.SELL, 40_000))
        position = broker.positions()["EUR/USD"]
        assert position.qty == pytest.approx(60_000)
        assert position.side is Side.BUY

    def test_flip_reverses_the_position(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 50_000))
        broker.submit(Order("EUR/USD", Side.SELL, 100_000))
        position = broker.positions()["EUR/USD"]
        assert position.side is Side.SELL
        assert position.qty == pytest.approx(50_000)
        assert len(broker.closed_trades()) == 1


class TestFifoAccounting:
    def test_partial_exits_use_the_tranche_that_was_actually_sold(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 50_000, tag="first"))
        broker.on_candle(make_candle("EUR/USD", 1.09000, index=1))
        broker.submit(Order("EUR/USD", Side.BUY, 50_000, tag="second"))
        first_entry = 1.10000 + 0.2 * 0.0001 + 0.5 * 0.0001
        second_entry = 1.09000 + 0.2 * 0.0001 + 0.5 * 0.0001

        # FIFO: selling 50k closes the *first* tranche only
        broker.submit(Order("EUR/USD", Side.SELL, 50_000, reduce_only=True))
        assert broker.closed_trades()[0].entry_price == pytest.approx(first_entry)

        # selling the rest closes the second tranche
        broker.submit(Order("EUR/USD", Side.SELL, 50_000, reduce_only=True))
        assert broker.closed_trades()[1].entry_price == pytest.approx(second_entry)

    def test_scale_out_reports_each_tranche_separately(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 50_000))
        broker.on_candle(make_candle("EUR/USD", 1.09000, index=1))
        broker.submit(Order("EUR/USD", Side.BUY, 50_000))
        # sell the first tranche only
        broker.submit(Order("EUR/USD", Side.SELL, 50_000, reduce_only=True))
        trades = broker.closed_trades()
        assert len(trades) == 1
        assert trades[0].entry_price < 1.1001
        assert broker.positions()["EUR/USD"].qty == pytest.approx(50_000)

    def test_reduce_only_without_a_position_is_a_noop(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        assert broker.submit(Order("EUR/USD", Side.SELL, 100_000, reduce_only=True)) is None

    def test_reduce_only_with_the_wrong_side_is_a_noop(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000))
        # a BUY reduce-only against a long cannot reduce anything
        assert broker.submit(Order("EUR/USD", Side.BUY, 10_000, reduce_only=True)) is None


class TestCommissionAndEquity:
    def test_commission_is_charged_per_side(self):
        broker = make_broker(commission_per_million=10.0)
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        fill = broker.submit(Order("EUR/USD", Side.BUY, 100_000))
        # 100k units x 1.10007 = 0.110007M notional -> 0.110007 x 10
        assert fill.commission == pytest.approx(0.110007 * 10.0)
        assert broker.cash() == pytest.approx(100_000 - 1.10007)

    def test_equity_tracks_unrealised_pnl(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000))
        # bought at ask + slippage, marked at mid -> down by half the spread + slippage
        entry_cost = 100_000 * (0.5 * 0.0001 + 0.2 * 0.0001)
        assert broker.equity() == pytest.approx(100_000.0 - entry_cost)
        broker.on_candle(make_candle("EUR/USD", 1.11000, index=1))
        assert broker.equity() > 100_000.0
        assert broker.unrealized_pnl() > 0

    def test_realised_pnl_lands_in_cash(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(Order("EUR/USD", Side.BUY, 100_000, take_profit=1.1200))
        broker.on_candle(make_candle("EUR/USD", 1.1150, index=1, high=1.1250, low=1.1100))
        assert broker.cash() > 100_000.0
        assert broker.equity() == pytest.approx(broker.cash())


class TestRestingOrders:
    def test_limit_buy_fills_when_price_trades_through(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        assert broker.submit(
            Order("EUR/USD", Side.BUY, 10_000, order_type=OrderType.LIMIT, limit_price=1.0950)
        ) is None
        assert len(broker.working_orders()) == 1
        broker.on_candle(make_candle("EUR/USD", 1.0980, index=1, low=1.0940, high=1.0990))
        assert broker.working_orders() == []
        assert broker.positions()["EUR/USD"].qty == pytest.approx(10_000)

    def test_limit_buy_gaps_through_and_fills_better(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(
            Order("EUR/USD", Side.BUY, 10_000, order_type=OrderType.LIMIT, limit_price=1.0950)
        )
        # opens below the limit -> fill at the open, which is a better price
        broker.on_candle(make_candle("EUR/USD", 1.0920, index=1, low=1.0910, high=1.0940))
        assert broker.positions()["EUR/USD"].avg_price < 1.0950

    def test_limit_sell_fills_when_price_rises(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(
            Order("EUR/USD", Side.SELL, 10_000, order_type=OrderType.LIMIT, limit_price=1.1050)
        )
        broker.on_candle(make_candle("EUR/USD", 1.1030, index=1, low=1.1020, high=1.1060))
        assert broker.positions()["EUR/USD"].side is Side.SELL

    def test_stop_buy_triggers_upward(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(
            Order("EUR/USD", Side.BUY, 10_000, order_type=OrderType.STOP, stop_price=1.1050)
        )
        broker.on_candle(make_candle("EUR/USD", 1.1030, index=1, low=1.1020, high=1.1060))
        assert broker.positions()["EUR/USD"].qty == pytest.approx(10_000)

    def test_cancel_orders(self, broker):
        broker.on_candle(make_candle("EUR/USD", 1.10000))
        broker.submit(
            Order("EUR/USD", Side.BUY, 10_000, order_type=OrderType.LIMIT, limit_price=1.0900)
        )
        assert broker.cancel_orders("EUR/USD") == 1
        assert broker.working_orders() == []
