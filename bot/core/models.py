"""Core value objects shared by the trading bot, the scaling bot and the backtester.

Everything in this package is plain stdlib Python on purpose: the bot has to run
on a laptop, a VPS or inside a container without a heavy dependency tree.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# Enums / small helpers
# --------------------------------------------------------------------------- #


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short.  Makes the PnL maths direction agnostic."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def round_to_lot(qty: float, lot: float = 1_000.0) -> float:
    """Round a quantity down to a whole number of ``lot`` sized units."""
    if qty <= 0 or lot <= 0:
        return 0.0
    if qty < lot:
        return lot  # never round a real position away to zero
    return math.floor(qty / lot) * lot


# --------------------------------------------------------------------------- #
# Symbol metadata
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SymbolSpec:
    """FX pair metadata.

    ``pip_size``  price increment of one pip (0.0001 for most pairs, 0.01 for JPY pairs)
    ``contract``  units of the base currency represented by 1.0 of ``qty``
    """

    symbol: str
    base_ccy: str
    quote_ccy: str
    pip_size: float = 0.0001
    contract: float = 1.0

    @property
    def is_usd_quoted(self) -> bool:
        """EUR/USD style: the pip value is already in USD."""
        return self.quote_ccy == "USD"

    @property
    def is_usd_based(self) -> bool:
        """USD/JPY style: profit is quoted in the counter currency."""
        return self.base_ccy == "USD"

    def pips(self, price_distance: float) -> float:
        return abs(price_distance) / self.pip_size

    def unit_value(self, price: float, account_ccy: str = "USD") -> float:
        """What one unit of the base currency costs in the account currency.

        EUR/USD at 1.10 -> 1.10 USD per EUR.  USD/JPY at 150 -> 1 USD per USD.
        Crosses with no account-currency leg are approximated by the pair's own
        rate (there is no rate available to convert with).
        """
        if self.quote_ccy == account_ccy:
            return price
        if self.base_ccy == account_ccy:
            return 1.0
        return price

    def notional(self, qty: float, price: float, account_ccy: str = "USD") -> float:
        """Position size expressed in the account currency."""
        return qty * self.unit_value(price, account_ccy)

    def pnl(
        self,
        qty: float,
        entry: float,
        exit: float,
        side: Side,
        account_ccy: str = "USD",
    ) -> float:
        """Profit/loss in ``account_ccy`` for a closed position of ``qty`` units.

        USD-quoted pairs convert 1:1.  USD-based pairs (USD/JPY, USD/CHF) quote
        the result in the counter currency so we divide by the exit rate.  Crosses
        with no USD leg are approximated with the exit rate - good enough for
        risk sizing and reporting.
        """
        raw = qty * (exit - entry) * side.sign
        if self.quote_ccy == account_ccy:
            return raw
        if self.base_ccy == account_ccy:
            return raw / exit if exit else 0.0
        return raw / exit if exit else 0.0


_SPEC_CACHE: Dict[str, SymbolSpec] = {}


def spec_for(symbol: str) -> SymbolSpec:
    """Build (and cache) a :class:`SymbolSpec` from a pair string like ``EUR/USD``."""
    key = symbol.upper().strip().replace("_", "/").replace("-", "/")
    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]
    sym = key
    base, _, quote = sym.partition("/")
    if not quote:  # "EURUSD" -> "EUR" / "USD"
        base, quote = sym[:3], sym[3:]
    base, quote = base or "XXX", quote or "YYY"
    pip = 0.01 if "JPY" in (base, quote) else 0.0001
    spec = SymbolSpec(symbol=f"{base}/{quote}", base_ccy=base, quote_ccy=quote, pip_size=pip)
    _SPEC_CACHE[key] = spec
    return spec


def stop_for_risk(
    spec: SymbolSpec,
    side: Side,
    avg_price: float,
    qty: float,
    risk_amount: float,
    account_ccy: str = "USD",
    iterations: int = 48,
) -> float:
    """Price at which a position of ``qty`` @ ``avg_price`` loses ``risk_amount``.

    Solved by bisection so it stays correct for USD-quoted, USD-based and cross
    pairs without a special case per quoting convention.
    """
    if qty <= 0 or risk_amount <= 0 or avg_price <= 0:
        return avg_price

    def loss_at(distance: float) -> float:
        stop = avg_price - side.sign * distance
        return -spec.pnl(qty, avg_price, stop, side, account_ccy)

    lo, hi = 0.0, avg_price  # a 100% adverse move is a safe upper bound
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if loss_at(mid) < risk_amount:
            lo = mid
        else:
            hi = mid
    distance = 0.5 * (lo + hi)
    return avg_price - side.sign * distance


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candle:
    symbol: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError(f"inconsistent OHLC for {self.symbol}: {self}")

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class Quote:
    symbol: str
    time: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


# --------------------------------------------------------------------------- #
# Orders / fills / positions / trades
# --------------------------------------------------------------------------- #

_order_ids = itertools.count(1)


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reduce_only: bool = False
    tag: str = ""
    id: int = field(default_factory=lambda: next(_order_ids))

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("order qty must be positive")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order needs a limit_price")
        if self.order_type is OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP order needs a stop_price")


@dataclass(frozen=True)
class Fill:
    order_id: int
    symbol: str
    side: Side
    qty: float
    price: float
    time: datetime
    commission: float = 0.0
    slippage: float = 0.0
    tag: str = ""
    reduce_only: bool = False

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Position:
    """A netted position.  ``tranches`` records every scale-in entry."""

    symbol: str
    side: Side
    qty: float = 0.0
    avg_price: float = 0.0
    opened_at: datetime = field(default_factory=utcnow)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    initial_qty: float = 0.0
    realized_pnl: float = 0.0
    tranches: list[tuple[float, float]] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.qty > 1e-9

    @property
    def notional(self) -> float:
        """Raw quote-currency notional (qty x price)."""
        return self.qty * self.avg_price

    def account_notional(self, account_ccy: str = "USD") -> float:
        """Notional converted into the account currency."""
        return spec_for(self.symbol).notional(self.qty, self.avg_price, account_ccy)

    @property
    def adds(self) -> int:
        """Number of scale-in tranches beyond the first."""
        return max(len(self.tranches) - 1, 0)

    def apply_fill(self, fill: Fill) -> None:
        if fill.side is not self.side:
            raise ValueError("cannot apply an opposite-side fill to a position")
        if not self.is_open:
            self.side = fill.side
            self.opened_at = fill.time
            self.initial_qty = fill.qty
            self.avg_price = fill.price
            self.qty = fill.qty
        else:
            total = self.qty + fill.qty
            self.avg_price = (self.avg_price * self.qty + fill.price * fill.qty) / total
            self.qty = total
        self.tranches.append((fill.price, fill.qty))

    def unrealized(self, price: float, spec: SymbolSpec, account_ccy: str = "USD") -> float:
        if not self.is_open:
            return 0.0
        return spec.pnl(self.qty, self.avg_price, price, self.side, account_ccy)


@dataclass
class Trade:
    """A completed round trip (entry -> exit), including scaled entries/exits."""

    symbol: str
    side: Side
    qty: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float = 0.0
    exit_reason: str = ""
    tags: list[str] = field(default_factory=list)
    max_r: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.commission

    @property
    def duration_minutes(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60.0


# --------------------------------------------------------------------------- #
# Strategy output
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Signal:
    """What a strategy emits when it wants to be in the market."""

    symbol: str
    side: Side
    strength: float = 1.0                  # 0..1 conviction, scales the position size
    stop_distance: Optional[float] = None  # price distance to the protective stop
    take_profit: Optional[float] = None
    reason: str = ""
