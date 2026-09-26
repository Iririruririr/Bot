"""Grid / DCA ladder bot - the other flavour of "scaling bot".

Where :mod:`bot.scaling.engine` manages *one* position by scaling in and out of
it, this module places a **ladder** of resting limit orders around a price and
lets the market scale *it* into the position as it trades through each rung:

    buy limits   at centre - k * spacing   -> each rung carries its own take-profit
    sell limits  at centre + k * spacing   -> shorts the range symmetrically

Classic FX grid trading: it harvests small profits while price oscillates and
bleeds in a strong trend, so always pair it with a hard stop or a bounded
notional (``max_total_qty``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from bot.brokers.paper import PaperBroker
from bot.core.models import Candle, Order, OrderType, Side, spec_for


@dataclass
class LadderConfig:
    centre_price: float
    levels: int = 5                   # rungs on each side of the centre
    spacing_pips: float = 20.0        # distance between rungs
    qty_per_level: float = 10_000.0   # base units per rung
    take_profit_pips: float = 15.0    # profit target per rung
    direction: str = "both"           # "buy" | "sell" | "both"
    rearm: bool = True                # re-place a rung once its target fills
    max_total_qty: float = 0.0        # 0 = unlimited
    stop_loss_pips: float = 0.0       # optional hard stop on the whole position

    def __post_init__(self) -> None:
        if self.direction not in ("buy", "sell", "both"):
            raise ValueError("direction must be buy, sell or both")
        if self.levels < 1 or self.spacing_pips <= 0:
            raise ValueError("levels >= 1 and spacing_pips > 0 required")


@dataclass
class Rung:
    """One rung of the ladder and the orders attached to it."""

    price: float
    side: Side
    qty: float
    entry: Optional[Order] = None
    target: Optional[Order] = None
    filled_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self.target is not None:
            return "target_resting"
        if self.filled_at is not None:
            return "filled"
        return "open"


class GridLadder:
    """Places and maintains a ladder of resting limit orders."""

    def __init__(
        self,
        config: LadderConfig,
        broker: PaperBroker,
        symbol: str = "EUR/USD",
        on_event=None,
    ):
        self.config = config
        self.broker = broker
        self.symbol_name = symbol
        self.spec = spec_for(symbol)
        self.on_event = on_event
        self.rungs: List[Rung] = []
        self.stopped = False   # set once the hard stop flattens the ladder
        self.stats = {"fills": 0, "targets_hit": 0, "rearmed": 0, "blocked": 0}

    # ------------------------------------------------------------------ build --
    def rung_prices(self) -> List[Tuple[float, Side]]:
        """Every resting price with the side that trades there."""
        step = self.config.spacing_pips * self.spec.pip_size
        out: List[Tuple[float, Side]] = []
        for k in range(1, self.config.levels + 1):
            if self.config.direction in ("buy", "both"):
                out.append((self.config.centre_price - k * step, Side.BUY))
            if self.config.direction in ("sell", "both"):
                out.append((self.config.centre_price + k * step, Side.SELL))
        return out

    def build(self) -> List[Rung]:
        """Place the initial ladder and return the rungs."""
        self.rungs = []
        for price, side in self.rung_prices():
            rung = Rung(price=price, side=side, qty=self.config.qty_per_level)
            rung.entry = Order(
                symbol=self.symbol_name,
                side=side,
                qty=self.config.qty_per_level,
                order_type=OrderType.LIMIT,
                limit_price=price,
                tag=f"grid@{price:.5f}",
            )
            self.broker.submit(rung.entry)
            self.rungs.append(rung)
            self._emit("grid_placed", price=price, side=side.value, qty=rung.qty)
        return self.rungs

    # ---------------------------------------------------------------- runtime --
    def on_candle(self, candle: Candle) -> None:
        """Track rung fills, arm take-profits, re-arm closed rungs.

        Runs in two phases: the resting book is re-read between them so a
        take-profit placed *this* bar is never mistaken for one that filled.
        """
        self._check_entries(candle)
        self._check_targets(candle)

        if self.config.stop_loss_pips:
            self._check_hard_stop(candle)

    def _check_entries(self, candle: Candle) -> None:
        if self.stopped:
            return
        resting = {id(order) for order in self.broker.working_orders()}
        for rung in self.rungs:
            if rung.filled_at is not None or rung.entry is None:
                continue
            if id(rung.entry) in resting:
                continue
            rung.filled_at = candle.close
            self.stats["fills"] += 1
            self._emit("grid_fill", price=rung.price, side=rung.side.value, qty=rung.qty)
            self._arm_target(rung, candle)

    def _check_targets(self, candle: Candle) -> None:
        if self.stopped:
            return
        resting = {id(order) for order in self.broker.working_orders()}
        for rung in self.rungs:
            if rung.target is None or id(rung.target) in resting:
                continue
            self.stats["targets_hit"] += 1
            self._emit("grid_target_hit", price=rung.target.limit_price, qty=rung.qty)
            rung.target = None
            rung.filled_at = None
            if self.config.rearm:
                self.stats["rearmed"] += 1
                self._place_entry(rung)

    def _place_entry(self, rung: Rung) -> None:
        if not self._exposure_allows():
            self.stats["blocked"] += 1
            self._emit("grid_blocked", price=rung.price, reason="max_total_qty")
            return
        rung.entry = Order(
            symbol=self.symbol_name,
            side=rung.side,
            qty=rung.qty,
            order_type=OrderType.LIMIT,
            limit_price=rung.price,
            tag=f"grid@{rung.price:.5f}",
        )
        self.broker.submit(rung.entry)

    def _arm_target(self, rung: Rung, candle: Candle) -> None:
        """Anchor the take-profit on the rung price, not on the live price.

        The real fill for a buy limit is ``min(open, limit)``, so anchoring on the
        rung price is slightly conservative and, more importantly, deterministic.
        """
        offset = self.config.take_profit_pips * self.spec.pip_size
        target_price = (
            rung.price + offset if rung.side is Side.BUY else rung.price - offset
        )
        if rung.side is Side.BUY and target_price <= candle.close:
            return  # already through the target - never sell below the market
        if rung.side is Side.SELL and target_price >= candle.close:
            return
        rung.target = Order(
            symbol=self.symbol_name,
            side=rung.side.opposite,
            qty=rung.qty,
            order_type=OrderType.LIMIT,
            limit_price=target_price,
            reduce_only=True,
            tag=f"grid_tp@{target_price:.5f}",
        )
        self.broker.submit(rung.target)
        self._emit("grid_target", price=target_price, qty=rung.qty)

    def _exposure_allows(self) -> bool:
        if not self.config.max_total_qty:
            return True
        open_qty = sum(
            rung.qty for rung in self.rungs if rung.filled_at is not None and rung.target is not None
        )
        return open_qty + self.config.qty_per_level <= self.config.max_total_qty

    def _check_hard_stop(self, candle: Candle) -> None:
        """Close everything if the pair moves ``stop_loss_pips`` against centre."""
        if self.stopped:
            return
        distance = abs(candle.close - self.config.centre_price) / self.spec.pip_size
        if distance < self.config.stop_loss_pips:
            return
        self.stopped = True
        self.broker.cancel_orders(self.symbol_name)
        for rung in self.rungs:
            # drop the stale order references: a cancelled order that is no
            # longer resting must not be mistaken for a fresh fill
            rung.entry = None
            rung.target = None
            rung.filled_at = None
        position = self.broker.positions().get(self.symbol_name)
        if position is not None and position.is_open:
            self.broker.submit(
                Order(
                    symbol=self.symbol_name,
                    side=position.side.opposite,
                    qty=position.qty,
                    order_type=OrderType.MARKET,
                    reduce_only=True,
                    tag="grid_hard_stop",
                )
            )
        self._emit("grid_hard_stop", price=candle.close, pips=distance)

    # ---------------------------------------------------------------- reporting
    def summary(self) -> Dict[str, object]:
        open_rungs = sum(1 for r in self.rungs if r.state == "open")
        filled = sum(1 for r in self.rungs if r.state == "filled")
        with_target = sum(1 for r in self.rungs if r.state == "target_resting")
        return {
            "rungs": len(self.rungs),
            "open": open_rungs,
            "filled": filled,
            "target_resting": with_target,
            "stopped": self.stopped,
            **self.stats,
        }

    def _emit(self, kind: str, **payload) -> None:
        if self.on_event:
            self.on_event(kind, symbol=self.symbol_name, **payload)
