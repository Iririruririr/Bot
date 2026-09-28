"""The scaling bot - position management for entries and exits.

This is the second half of the system.  The *trading bot* decides **when** to be
in the market; the *scaling bot* decides **how much** to have on and **when** to
get out, by:

* **scale-in** - adding to a winner (pyramiding) or averaging into a pullback
  (DCA) in equal tranches, one add per ``add_step_r`` of price movement
* **scale-out** - closing fixed fractions of the position at R multiples
  (e.g. 50% at 1R, 25% at 2R, the rest rides)
* **risk control** - after every add the stop is re-solved so the *total* open
  risk stays constant instead of growing with each tranche

Everything is expressed in **R** (the initial risk unit = distance from entry to
the first stop), which keeps the config meaningful across pairs and timeframes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bot.core.models import (
    Candle,
    Order,
    OrderType,
    Position,
    Side,
    spec_for,
    stop_for_risk,
)
from bot.strategies.indicators import atr as atr_series

EventHook = Callable[[str, dict], None]


@dataclass(frozen=True)
class ScaleOutLeg:
    """Close ``fraction`` of the original position once price reaches ``r`` R."""

    r: float
    fraction: float

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("scale-out r must be positive")
        if not 0 < self.fraction <= 1:
            raise ValueError("scale-out fraction must be in (0, 1]")


@dataclass
class ScalingConfig:
    """Configuration for the scaling bot.

    ``tranches``      total entries, including the first (1 = no scale-in)
    ``add_step_r``    price movement (in R) between successive adds
    ``add_size_mult`` size of each add relative to the initial tranche
    ``mode``          ``pyramid`` adds into strength, ``dca`` adds into weakness,
                      ``off`` disables scale-in
    ``scale_out``     partial-exit ladder, evaluated every bar
    ``breakeven_at_r`` move the stop to entry once this R is reached
    ``trail_atr_mult`` chandelier trail distance in ATRs (None disables)
    ``trail_start_r``  R that must be reached before trailing activates
    ``time_stop_bars`` close the position after this many bars (None disables)
    """

    tranches: int = 1
    add_step_r: float = 1.0
    add_size_mult: float = 1.0
    mode: str = "pyramid"
    scale_out: Tuple[ScaleOutLeg, ...] = ()
    breakeven_at_r: Optional[float] = None
    trail_atr_mult: Optional[float] = None
    trail_start_r: float = 1.0
    max_add_exposure_pct: float = 100.0  # skip adds that would breach this
    time_stop_bars: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in ("pyramid", "dca", "off"):
            raise ValueError(f"unknown scaling mode {self.mode!r}")
        if self.tranches < 1:
            raise ValueError("tranches must be >= 1")
        if self.scale_out:
            total = sum(leg.fraction for leg in self.scale_out)
            if total > 1.0 + 1e-9:
                raise ValueError("scale-out fractions sum to more than the position")
        # keep the ladder sorted so the engine can walk it in order
        self.scale_out = tuple(sorted(self.scale_out, key=lambda leg: leg.r))

    # ----------------------------------------------------------- factories --
    @classmethod
    def pyramid(cls, tranches: int = 3, step_r: float = 1.0, **kwargs) -> "ScalingConfig":
        """Add into strength: +1 tranche per ``step_r`` in our favour."""
        legs = kwargs.pop(
            "scale_out",
            (ScaleOutLeg(1.0, 0.5), ScaleOutLeg(2.0, 0.25), ScaleOutLeg(3.0, 0.25)),
        )
        return cls(tranches=tranches, add_step_r=step_r, mode="pyramid", scale_out=legs, **kwargs)

    @classmethod
    def dca(cls, tranches: int = 3, step_r: float = 0.5, **kwargs) -> "ScalingConfig":
        """Average into weakness: +1 tranche per ``step_r`` against us."""
        legs = kwargs.pop("scale_out", (ScaleOutLeg(1.0, 0.5), ScaleOutLeg(2.0, 0.5)))
        return cls(tranches=tranches, add_step_r=step_r, mode="dca", scale_out=legs, **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "ScalingConfig":
        data = dict(data or {})
        legs = data.pop("scale_out", ()) or ()
        scale_out = tuple(
            leg if isinstance(leg, ScaleOutLeg) else ScaleOutLeg(float(leg[0]), float(leg[1]))
            for leg in legs
        )
        return cls(scale_out=scale_out, **data)

    def to_dict(self) -> dict:
        return {
            "tranches": self.tranches,
            "add_step_r": self.add_step_r,
            "add_size_mult": self.add_size_mult,
            "mode": self.mode,
            "scale_out": [(leg.r, leg.fraction) for leg in self.scale_out],
            "breakeven_at_r": self.breakeven_at_r,
            "trail_atr_mult": self.trail_atr_mult,
            "trail_start_r": self.trail_start_r,
            "time_stop_bars": self.time_stop_bars,
        }


@dataclass
class ManagedPosition:
    """Everything the scaling bot remembers about one open position."""

    symbol: str
    side: Side
    first_entry: float
    initial_stop: float
    initial_qty: float
    adds_done: int = 0
    bars_held: int = 0
    taken: List[float] = field(default_factory=list)  # scale-out R levels already used
    breakeven_done: bool = False
    trailing: bool = False
    peak: float = 0.0
    trough: float = float("inf")   # +inf so the first price always becomes the trough
    max_progress_r: float = 0.0
    min_progress_r: float = 0.0
    current_atr: float = 0.0

    @property
    def r(self) -> float:
        """The initial risk unit, in price."""
        return abs(self.first_entry - self.initial_stop)

    def progress(self, price: float) -> float:
        """Where ``price`` sits in R units (+ = in profit)."""
        if self.r <= 0:
            return 0.0
        return (price - self.first_entry) * self.side.sign / self.r

    def update_extremes(self, price: float) -> None:
        self.peak = max(self.peak, price)
        self.trough = min(self.trough, price)
        self.max_progress_r = max(self.max_progress_r, self.progress(price))
        self.min_progress_r = min(self.min_progress_r, self.progress(price))


class ScalingEngine:
    """Manages open positions on behalf of the trading bot (or standalone)."""

    def __init__(
        self,
        config: Optional[ScalingConfig] = None,
        broker=None,
        account_ccy: str = "USD",
        on_event: Optional[EventHook] = None,
        atr_period: int = 14,
        risk=None,
    ):
        self.config = config or ScalingConfig()
        self.broker = broker
        self.account_ccy = account_ccy
        self.on_event = on_event
        self.atr_period = atr_period
        self.risk = risk
        self.managed: Dict[str, ManagedPosition] = {}
        self.stats = {
            "adds": 0,
            "scale_outs": 0,
            "breakevens": 0,
            "trail_updates": 0,
            "skipped_adds": 0,
        }

    # ------------------------------------------------------------------ api --
    def emit(self, kind: str, **payload) -> None:
        """Event hook signature matches :meth:`bot.core.engine.Engine.emit`."""
        if self.on_event:
            self.on_event(kind, **payload)

    def attach(self, symbol: str, position: Position) -> ManagedPosition:
        """Start managing an existing position, anchoring R on its stop."""
        stop = position.stop_loss
        if stop is None:
            # no stop supplied -> anchor 1% of price away so R stays well defined
            stop = position.avg_price * (1 - 0.01 * position.side.sign)
        mp = ManagedPosition(
            symbol=symbol,
            side=position.side,
            first_entry=position.avg_price,
            initial_stop=stop,
            initial_qty=position.initial_qty or position.qty,
            peak=position.avg_price,
            trough=position.avg_price,
        )
        self.managed[symbol] = mp
        self.emit("attach", symbol=symbol, entry=position.avg_price, stop=stop, r=mp.r)
        return mp

    def is_managing(self, symbol: str) -> bool:
        return symbol in self.managed

    def release(self, symbol: str) -> None:
        self.managed.pop(symbol, None)

    # -------------------------------------------------------------- runtime --
    def on_candle(self, candle: Candle, history: Optional[Sequence[Candle]] = None) -> List[Order]:
        """Evaluate scale-in / scale-out / stop management for the open position."""
        symbol = candle.symbol
        positions = self.broker.positions() if self.broker else {}
        position = positions.get(symbol)
        if position is None or not position.is_open:
            if symbol in self.managed:
                self.release(symbol)
            return []

        mp = self.managed.get(symbol)
        if mp is None:
            mp = self.attach(symbol, position)
        mp.bars_held += 1

        mp.current_atr = self._atr(history, candle)
        mp.update_extremes(candle.close)
        orders = self._decide(mp, position, candle)
        return orders

    def _atr(self, history: Optional[Sequence[Candle]], candle: Candle) -> float:
        if not history:
            return 0.0
        series = atr_series(
            [c.high for c in history],
            [c.low for c in history],
            [c.close for c in history],
            self.atr_period,
        )
        return series[-1] if series else 0.0

    # -------------------------------------------------------------- decisions
    def _decide(self, mp: ManagedPosition, position: Position, candle: Candle) -> List[Order]:
        orders: List[Order] = []
        price = candle.close
        r = mp.r
        if r <= 0:
            return orders
        progress = mp.progress(price)

        # 0) time stop ------------------------------------------------------- #
        if self.config.time_stop_bars and mp.bars_held >= self.config.time_stop_bars:
            self.emit("time_stop", symbol=mp.symbol, bars=mp.bars_held, price=price)
            self.release(mp.symbol)
            return [
                Order(
                    symbol=mp.symbol,
                    side=mp.side.opposite,
                    qty=position.qty,
                    order_type=OrderType.MARKET,
                    reduce_only=True,
                    tag="time_stop",
                )
            ]

        # 1) scale out ------------------------------------------------------- #
        for leg in self.config.scale_out:
            if leg.r in mp.taken:
                continue
            if progress < leg.r:
                continue
            qty = min(position.initial_qty * leg.fraction, position.qty)
            if qty <= 1e-9:
                continue
            orders.append(
                Order(
                    symbol=mp.symbol,
                    side=mp.side.opposite,
                    qty=qty,
                    order_type=OrderType.MARKET,
                    reduce_only=True,
                    tag=f"scaleout@{leg.r:g}R",
                )
            )
            mp.taken.append(leg.r)
            self.stats["scale_outs"] += 1
            self.emit(
                "scale_out",
                symbol=mp.symbol,
                r=leg.r,
                qty=qty,
                price=price,
                remaining=position.qty - qty,
            )

        # 2) breakeven ------------------------------------------------------- #
        if (
            self.config.breakeven_at_r is not None
            and not mp.breakeven_done
            and progress >= self.config.breakeven_at_r
        ):
            position.stop_loss = mp.first_entry
            mp.breakeven_done = True
            self.stats["breakevens"] += 1
            self.emit("breakeven", symbol=mp.symbol, stop=mp.first_entry, price=price)

        # 3) trailing -------------------------------------------------------- #
        if self.config.trail_atr_mult is not None and mp.current_atr > 0:
            if progress >= self.config.trail_start_r:
                mp.trailing = True
            if mp.trailing:
                trail_distance = self.config.trail_atr_mult * mp.current_atr
                # longs trail below the highest high, shorts above the lowest low
                candidate = (
                    mp.peak - trail_distance if mp.side is Side.BUY else mp.trough + trail_distance
                )
                current = position.stop_loss
                improves = current is None or (
                    candidate > current if mp.side is Side.BUY else candidate < current
                )
                if improves and self._is_beyond_entry(mp, candidate):
                    position.stop_loss = candidate
                    self.stats["trail_updates"] += 1
                    self.emit("trail", symbol=mp.symbol, stop=candidate, price=price)

        # 4) scale in -------------------------------------------------------- #
        if self.config.tranches > 1 and self.config.mode != "off":
            if mp.adds_done < self.config.tranches - 1:
                orders.extend(self._maybe_add(mp, position, candle, progress))

        return orders

    def _is_beyond_entry(self, mp: ManagedPosition, candidate: float) -> bool:
        """A trailing stop must never sit on the wrong side of the entry price."""
        if mp.side is Side.BUY:
            return candidate > mp.first_entry
        return candidate < mp.first_entry

    def _maybe_add(
        self, mp: ManagedPosition, position: Position, candle: Candle, progress: float
    ) -> List[Order]:
        cfg = self.config
        step = (mp.adds_done + 1) * cfg.add_step_r
        if cfg.mode == "pyramid":
            due = progress >= step
        else:  # dca
            due = -progress >= step
        if not due:
            return []

        desired = mp.initial_qty * cfg.add_size_mult
        if desired <= 1e-9:
            return []

        # trim the add to whatever exposure budget is left, so a pyramid still
        # builds into a strong move instead of being blocked outright
        headroom = self._exposure_headroom(mp)
        add_qty = min(desired, headroom)
        if add_qty <= 1e-9:
            self.stats["skipped_adds"] += 1
            self.emit("add_skipped", symbol=mp.symbol, reason="exposure cap", price=candle.close)
            return []

        mp.adds_done += 1
        self.stats["adds"] += 1
        self.emit(
            "scale_in",
            symbol=mp.symbol,
            tranche=mp.adds_done + 1,
            qty=add_qty,
            price=candle.close,
            progress_r=progress,
        )
        return [
            Order(
                symbol=mp.symbol,
                side=mp.side,
                qty=add_qty,
                order_type=OrderType.MARKET,
                tag=f"scalein#{mp.adds_done + 1}",
            )
        ]

    def _exposure_headroom(self, mp: ManagedPosition) -> float:
        """Largest add that still fits inside the exposure cap (0 if none)."""
        if not self.broker:
            return float("inf")
        equity = self.broker.equity()
        if equity <= 0:
            return 0.0
        if self.risk is not None:
            return self.risk.max_qty_for_exposure(
                mp.symbol, equity, self.broker.open_notional(), mp.first_entry
            )
        budget = equity * self.config.max_add_exposure_pct / 100.0 - self.broker.open_notional()
        return max(budget, 0.0) / spec_for(mp.symbol).unit_value(
            mp.first_entry, self.account_ccy
        )

    # ------------------------------------------------------- post-fill hooks --
    def on_scale_in_filled(self, symbol: str, fill_price: float, fill_qty: float) -> None:
        """Re-solve the stop so total open risk stays constant after an add."""
        if not self.broker:
            return
        position = self.broker.positions().get(symbol)
        mp = self.managed.get(symbol)
        if position is None or mp is None or not position.is_open:
            return

        initial_risk = abs(
            self.broker.spec(symbol).pnl(
                mp.initial_qty,
                mp.first_entry,
                mp.initial_stop,
                mp.side,
                self.account_ccy,
            )
        )
        if initial_risk <= 0:
            return
        new_stop = stop_for_risk(
            self.broker.spec(symbol),
            mp.side,
            position.avg_price,
            position.qty,
            initial_risk,
            self.account_ccy,
        )
        # safety: a stop on the wrong side of the fill would trigger instantly
        if mp.side is Side.BUY and new_stop >= fill_price:
            return
        if mp.side is Side.SELL and new_stop <= fill_price:
            return
        position.stop_loss = new_stop
        self.emit(
            "risk_rebalanced",
            symbol=symbol,
            stop=position.stop_loss,
            qty=position.qty,
            risk=initial_risk,
        )
