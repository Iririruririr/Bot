"""Paper (simulated) broker - the default execution venue for the FX bots.

Modelled closely enough to a real FX broker that strategy results carry over:

* every fill crosses the spread (buy @ ask, sell @ bid) plus configurable slippage
* protective stops / take-profits are evaluated against each bar's high and low,
  with the stop assumed to be hit first when both are touched in the same bar
  (the pessimistic assumption)
* opposite-side market orders net against the open position before flipping
* commission is charged per side, in account currency
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from bot.core.models import (
    Candle,
    Fill,
    Order,
    OrderType,
    Position,
    Quote,
    Side,
    Trade,
    spec_for,
    utcnow,
)


@dataclass
class BrokerConfig:
    starting_cash: float = 100_000.0
    account_ccy: str = "USD"
    spread_pips: float = 1.0        # typical retail EUR/USD spread
    slippage_pips: float = 0.2      # extra adverse price on every market order
    commission_per_million: float = 0.0  # account ccy per 1M notional, per side
    lot_size: float = 1_000.0       # smallest tradeable increment, in base units


class Broker(ABC):
    """Execution interface shared by the paper broker and the live OANDA broker."""

    @abstractmethod
    def quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def submit(self, order: Order) -> Optional[Fill]: ...

    @abstractmethod
    def positions(self) -> Dict[str, Position]: ...

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def closed_trades(self) -> List[Trade]: ...


class PaperBroker(Broker):
    """Deterministic in-memory broker driven by bar or tick data."""

    def __init__(self, config: Optional[BrokerConfig] = None, spec_cache: Optional[dict] = None):
        self.config = config or BrokerConfig()
        self._cash = self.config.starting_cash
        self._positions: Dict[str, Position] = {}
        self._trades: List[Trade] = []
        self._working: List[Order] = []
        self._last_mid: Dict[str, float] = {}
        self._last_time: Dict[str, datetime] = {}
        self._specs: Dict[str, object] = spec_cache if spec_cache is not None else {}

    # ---------------------------------------------------------------- specs --
    def spec(self, symbol: str):
        if symbol not in self._specs:
            self._specs[symbol] = spec_for(symbol)
        return self._specs[symbol]

    # --------------------------------------------------------------- prices --
    def update_price(self, symbol: str, bid: float, ask: float, when: Optional[datetime] = None) -> None:
        self._last_mid[symbol] = (bid + ask) / 2.0
        self._last_time[symbol] = when or utcnow()

    def on_candle(self, candle: Candle) -> List[Trade]:
        """Mark the book to the bar, run protective stops, then resting orders."""
        spec = self.spec(candle.symbol)
        half_spread = spec.pip_size * self.config.spread_pips / 2.0
        self.update_price(
            candle.symbol,
            candle.close - half_spread,
            candle.close + half_spread,
            candle.time,
        )
        closed = self._check_protective_orders(candle)
        closed.extend(self._check_working_orders(candle))
        return closed

    def working_orders(self) -> List[Order]:
        return list(self._working)

    def cancel_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel resting orders (all, or just one symbol)."""
        before = len(self._working)
        if symbol is None:
            self._working.clear()
        else:
            self._working = [o for o in self._working if o.symbol != symbol]
        return before - len(self._working)

    def quote(self, symbol: str) -> Quote:
        if symbol not in self._last_mid:
            raise KeyError(f"no price for {symbol} yet - feed the broker a candle or tick first")
        spec = self.spec(symbol)
        half_spread = spec.pip_size * self.config.spread_pips / 2.0
        mid = self._last_mid[symbol]
        return Quote(symbol=symbol, time=self._last_time.get(symbol, utcnow()), bid=mid - half_spread, ask=mid + half_spread)

    # --------------------------------------------------------------- orders --
    def _slippage(self, symbol: str) -> float:
        return self.spec(symbol).pip_size * self.config.slippage_pips

    def _commission(self, qty: float, price: float) -> float:
        if self.config.commission_per_million <= 0:
            return 0.0
        return abs(qty * price) / 1_000_000.0 * self.config.commission_per_million

    def submit(self, order: Order) -> Optional[Fill]:
        """Execute or rest an order.

        Market orders fill immediately at the adverse side of the spread.
        Limit and stop orders rest on the book and are filled by
        :meth:`on_candle` when the market trades through their price.
        """
        if order.order_type is OrderType.MARKET:
            return self._fill(order, self._exec_price(order, self.quote(order.symbol)), self.quote(order.symbol).time)
        self._working.append(order)
        return None

    def _fill(self, order: Order, price: float, when: datetime) -> Optional[Fill]:
        """Fill ``order`` at ``price``, netting against the open position first."""
        position = self._positions.get(order.symbol)

        # ---- reduce-only orders simply shrink the position ----------------- #
        if order.reduce_only:
            if position is None or not position.is_open or position.side is order.side:
                return None
            self._close_position(position, order.qty, price, when, order.tag or "reduce")
            return None

        qty = order.qty
        # ---- netting: close what we can, then open the remainder ------------ #
        if position is not None and position.is_open and position.side is not order.side:
            close_qty = min(position.qty, qty)
            self._close_position(position, close_qty, price, when, order.tag or "flip")
            qty -= close_qty  # note: _close_position has already reduced position.qty
            if qty <= 1e-9:
                return None

        commission = self._commission(qty, price)
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            time=when,
            commission=commission,
            slippage=self._slippage(order.symbol),
            tag=order.tag,
        )
        self._apply_fill(fill, order)
        self._cash -= commission
        return fill

    def _exec_price(self, order: Order, quote: Quote) -> float:
        """Adverse side of the spread plus slippage."""
        sign = order.side.sign
        raw = quote.ask if order.side is Side.BUY else quote.bid
        return raw + sign * self._slippage(order.symbol)

    def _apply_fill(self, fill: Fill, order: Optional[Order] = None) -> None:
        position = self._positions.get(fill.symbol)
        if position is None or not position.is_open:
            self._positions[fill.symbol] = Position(symbol=fill.symbol, side=fill.side)
            position = self._positions[fill.symbol]
            # a brand new position inherits the bracket from the opening order
            if order is not None:
                position.stop_loss = order.stop_loss
                position.take_profit = order.take_profit
        position.apply_fill(fill)

    def _close_position(
        self,
        position: Position,
        qty: float,
        price: float,
        when: datetime,
        reason: str,
    ) -> None:
        """Close ``qty`` of ``position`` at ``price`` using FIFO tranche accounting.

        Consuming tranches oldest-first keeps every partial exit honest: the
        trade reports the entry price of the units actually sold, and the
        position's average cost is recomputed from what is left.
        """
        spec = self.spec(position.symbol)
        qty = min(qty, position.qty)
        if qty <= 1e-9:
            return

        consumed_cost, consumed_qty, remaining = 0.0, 0.0, qty
        while remaining > 1e-9 and position.tranches:
            tranche_price, tranche_qty = position.tranches[0]
            take = min(tranche_qty, remaining)
            consumed_cost += take * tranche_price
            consumed_qty += take
            remaining -= take
            if take >= tranche_qty - 1e-9:
                position.tranches.pop(0)
            else:
                position.tranches[0] = (tranche_price, tranche_qty - take)

        entry_price = consumed_cost / consumed_qty if consumed_qty > 0 else position.avg_price
        pnl = spec.pnl(qty, entry_price, price, position.side, self.config.account_ccy)
        commission = self._commission(qty, price)
        position.realized_pnl += pnl
        position.qty -= qty
        self._cash += pnl - commission

        if position.tranches:
            left_qty = sum(q for _, q in position.tranches)
            position.avg_price = (
                sum(p * q for p, q in position.tranches) / left_qty if left_qty else position.avg_price
            )

        self._trades.append(
            Trade(
                symbol=position.symbol,
                side=position.side,
                qty=qty,
                entry_price=entry_price,
                exit_price=price,
                entry_time=position.opened_at,
                exit_time=when,
                pnl=pnl,
                commission=commission,
                exit_reason=reason,
            )
        )
        if position.qty <= 1e-9:
            position.qty = 0.0
            self._positions.pop(position.symbol, None)

    # ---------------------------------------------------- protective orders --
    def _check_protective_orders(self, candle: Candle) -> List[Trade]:
        """Run stop-loss / take-profit against the bar's high and low.

        When both levels sit inside the same bar the stop is assumed to trigger
        first - the pessimistic assumption, which keeps backtests honest.
        """
        position = self._positions.get(candle.symbol)
        if position is None or not position.is_open:
            return []
        stop, target = position.stop_loss, position.take_profit
        long_side = position.side is Side.BUY
        hit_stop = stop is not None and (candle.low <= stop if long_side else candle.high >= stop)
        hit_target = target is not None and (
            candle.high >= target if long_side else candle.low <= target
        )
        if not (hit_stop or hit_target):
            return []

        if hit_stop:
            # gapped straight through the stop -> fill at the open, not the stop
            gapped = candle.open < stop if long_side else candle.open > stop
            price = candle.open if gapped else stop
            reason = "stop_loss"
        else:
            price, reason = target, "take_profit"
        before = len(self._trades)
        self._close_position(position, position.qty, price, candle.time, reason)
        return self._trades[before:]

    # ---------------------------------------------------------- resting orders
    def _check_working_orders(self, candle: Candle) -> List[Trade]:
        """Fill resting limit/stop orders whose price the bar traded through.

        A gapped-open fills at the open (better than the resting price) rather
        than at the resting price, which is what a real exchange would do.
        """
        filled: List[Trade] = []
        for order in list(self._working):
            if order.symbol != candle.symbol:
                continue
            price = self._resting_fill_price(order, candle)
            if price is None:
                continue
            self._working.remove(order)
            before = len(self._trades)
            self._fill(order, price, candle.time)
            filled.extend(self._trades[before:])
        return filled

    def _resting_fill_price(self, order: Order, candle: Candle) -> Optional[float]:
        long_side = order.side is Side.BUY
        if order.order_type is OrderType.LIMIT:
            if long_side and candle.low <= order.limit_price:
                return min(candle.open, order.limit_price)
            if not long_side and candle.high >= order.limit_price:
                return max(candle.open, order.limit_price)
            return None
        if order.order_type is OrderType.STOP:
            if long_side and candle.high >= order.stop_price:
                return max(candle.open, order.stop_price)
            if not long_side and candle.low <= order.stop_price:
                return min(candle.open, order.stop_price)
            return None
        return None

    # ------------------------------------------------------------- reporting --
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def cash(self) -> float:
        return self._cash

    def mid(self, symbol: str) -> Optional[float]:
        """Last mid price seen for ``symbol``, or None before the first quote."""
        return self._last_mid.get(symbol)

    def closed_trades(self) -> List[Trade]:
        return list(self._trades)

    def open_notional(self) -> float:
        """Total open notional in the account currency."""
        return sum(
            p.account_notional(self.config.account_ccy) for p in self._positions.values()
        )

    def unrealized_pnl(self) -> float:
        total = 0.0
        for symbol, position in self._positions.items():
            if position.is_open and symbol in self._last_mid:
                total += position.unrealized(self._last_mid[symbol], self.spec(symbol), self.config.account_ccy)
        return total

    def equity(self) -> float:
        return self._cash + self.unrealized_pnl()
