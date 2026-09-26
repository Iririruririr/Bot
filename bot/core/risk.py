"""Risk management - the guard rails that keep the bots alive.

The risk manager owns three jobs:

1. **Sizing** - convert a stop distance into a position size so that a stopped
   out trade costs a fixed fraction of equity ("risk 1% per trade").
2. **Limits** - cap how many positions, how much notional and how much leverage
   may be open at once.
3. **Kill switches** - halt trading on a daily loss limit or an equity drawdown
   from the high-water mark.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from bot.core.models import Position, round_to_lot, spec_for


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0        # % of equity lost if the stop is hit
    max_daily_loss_pct: float = 3.0        # halt for the day beyond this
    max_drawdown_pct: float = 10.0         # halt for good beyond this (from HWM)
    max_open_exposure_pct: float = 20.0    # total notional / equity
    max_positions: int = 4                 # concurrent open positions
    max_leverage: float = 10.0             # notional / equity ceiling
    lot_size: float = 1_000.0              # base-currency units per lot
    account_ccy: str = "USD"


class RiskViolation(Exception):
    """Raised by :meth:`RiskManager.assert_can_open` in strict mode."""


class RiskManager:
    """Stateful risk gate consulted before every entry and every scale-in."""

    def __init__(self, config: Optional[RiskConfig] = None, starting_equity: float = 100_000.0):
        self.config = config or RiskConfig()
        self.starting_equity = starting_equity
        self.peak_equity = starting_equity
        self.halted = False
        self.halt_reason = ""
        self._day: Optional[str] = None
        self._day_start_equity = starting_equity
        self._day_realized = 0.0

    # ------------------------------------------------------------------ days --
    def _roll_day(self, now: datetime, equity: float) -> None:
        key = now.astimezone(timezone.utc).date().isoformat()
        if key != self._day:
            self._day = key
            self._day_start_equity = equity
            self._day_realized = 0.0

    def register_equity(self, equity: float, now: Optional[datetime] = None) -> None:
        """Feed current equity in every bar; updates drawdown and daily stats."""
        now = now or datetime.now(timezone.utc)
        self._roll_day(now, equity)
        self.peak_equity = max(self.peak_equity, equity)

        daily_pnl = equity - self._day_start_equity
        if daily_pnl <= -self.config.max_daily_loss_pct / 100.0 * self._day_start_equity:
            self._halt(f"daily loss limit hit ({daily_pnl:,.0f} on {self._day})")
            return
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100.0
        if drawdown >= self.config.max_drawdown_pct:
            self._halt(f"max drawdown hit ({drawdown:.1f}% from peak)")

    def _halt(self, reason: str) -> None:
        if not self.halted:
            self.halted = True
            self.halt_reason = reason

    def reset_daily(self, equity: float) -> None:
        self._day = None
        self._day_realized = 0.0
        self._day_start_equity = equity

    def unhalt(self) -> None:
        self.halted = False
        self.halt_reason = ""

    # --------------------------------------------------------------- gating --
    def can_open(self, equity: float, positions: Dict[str, Position], now: Optional[datetime] = None) -> Tuple[bool, str]:
        if self.halted:
            return False, f"halted: {self.halt_reason}"
        now = now or datetime.now(timezone.utc)
        self._roll_day(now, equity)
        if len(positions) >= self.config.max_positions:
            return False, f"max positions ({self.config.max_positions}) reached"
        notional = sum(p.account_notional(self.config.account_ccy) for p in positions.values())
        exposure_pct = notional / equity * 100.0 if equity > 0 else 100.0
        if exposure_pct >= self.config.max_open_exposure_pct:
            return False, f"exposure cap {exposure_pct:.1f}% >= {self.config.max_open_exposure_pct}%"
        if equity > 0 and notional / equity >= self.config.max_leverage:
            return False, f"leverage cap {notional / equity:.1f}x >= {self.config.max_leverage}x"
        daily_pnl = equity - self._day_start_equity
        if daily_pnl <= -self.config.max_daily_loss_pct / 100.0 * self._day_start_equity:
            return False, "daily loss limit reached"
        return True, ""

    def assert_can_open(self, equity: float, positions: Dict[str, Position]) -> None:
        ok, reason = self.can_open(equity, positions)
        if not ok:
            raise RiskViolation(reason)

    # --------------------------------------------------------------- sizing --
    def position_size(
        self,
        symbol: str,
        equity: float,
        entry: float,
        stop: Optional[float],
        strength: float = 1.0,
        tranches: int = 1,
    ) -> float:
        """Units to buy/sell so that a stop-out loses ``risk_per_trade_pct``.

        ``tranches`` splits the risk budget across the entries the scaling bot
        plans to make, so a 3-tranche pyramid risks a third per add instead of
        the full budget three times over.

        Returns 0 when no stop is supplied or the risk budget cannot be used.
        """
        if stop is None or entry <= 0:
            return 0.0
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0

        spec = spec_for(symbol)
        tranches = max(int(tranches or 1), 1)
        risk_amount = (
            equity * self.config.risk_per_trade_pct / 100.0 * max(0.0, min(strength, 1.0)) / tranches
        )
        if risk_amount <= 0:
            return 0.0

        # Solve for the quantity whose loss at the stop equals the risk budget.
        #   USD-quoted pair : loss = qty * stop_distance
        #   USD-based pair  : loss = qty * stop_distance / exit
        # A JPY stop distance of 0.20 on a 150.00 pair is ~0.13% of price, so the
        # USD-based branch multiplies by the entry rate to compensate.
        if spec.quote_ccy == self.config.account_ccy:
            qty = risk_amount / stop_distance
        elif spec.base_ccy == self.config.account_ccy:
            qty = risk_amount * entry / stop_distance
        else:
            qty = risk_amount * entry / stop_distance  # cross approximation

        # Hard account-level ceiling only: the portfolio exposure cap is applied
        # by the engine *after* sizing, so it can trim a position to fit rather
        # than silently shrink every trade below its risk target.
        unit_value = spec.unit_value(entry, self.config.account_ccy)
        if unit_value > 0:
            qty = min(qty, equity * self.config.max_leverage / unit_value)
        return round_to_lot(qty, self.config.lot_size)

    def max_qty_for_exposure(
        self, symbol: str, equity: float, current_notional: float, entry: float
    ) -> float:
        """Largest position (in base units) that still fits the exposure cap."""
        if equity <= 0 or entry <= 0:
            return 0.0
        budget = equity * self.config.max_open_exposure_pct / 100.0 - current_notional
        if budget <= 0:
            return 0.0
        unit_value = spec_for(symbol).unit_value(entry, self.config.account_ccy)
        if unit_value <= 0:
            return 0.0
        return round_to_lot(budget / unit_value, self.config.lot_size)

    # ------------------------------------------------------------- reporting --
    def daily_pnl(self, equity: float) -> float:
        return equity - self._day_start_equity

    def drawdown_pct(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity * 100.0
