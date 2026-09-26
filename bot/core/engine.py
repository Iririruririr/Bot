"""The trading engine - wires market data, strategies, risk, the scaling bot and
the broker into one deterministic bar-by-bar loop.

Execution model (identical in backtest, paper and live):

    for every closed bar:
        1. the broker marks to market and runs protective stops
        2. the scaling bot manages any open position (adds / partial exits / stops)
        3. each strategy looks for a new entry, sized by the risk manager
        4. an equity snapshot is recorded

Signals are evaluated on the **close** of a bar and filled at that same close,
crossing the spread plus slippage - no look-ahead, and no free money.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bot.core.models import Candle, Order, OrderType, Position, Trade
from bot.core.risk import RiskManager
from bot.scaling.engine import ScalingEngine
from bot.strategies.base import Strategy

EventHook = Callable[[str, dict], None]


@dataclass
class EngineConfig:
    exit_on_flip: bool = True       # close when a strategy signals the other way
    max_history_bars: int = 800     # rolling window kept per symbol
    cooldown_bars: int = 0          # bars to wait after an exit before re-entering
    record_equity: bool = True


class Engine:
    """Bar-driven trading engine."""

    def __init__(
        self,
        broker,
        strategies: Optional[Sequence[Strategy]] = None,
        risk: Optional[RiskManager] = None,
        scaling: Optional[ScalingEngine] = None,
        storage=None,
        config: Optional[EngineConfig] = None,
        on_event: Optional[EventHook] = None,
    ):
        self.broker = broker
        self.strategies: List[Strategy] = list(strategies or [])
        self.risk = risk
        self.scaling = scaling
        self.storage = storage
        self.config = config or EngineConfig()
        self.on_event = on_event

        self.history: Dict[str, List[Candle]] = defaultdict(list)
        self.bar_index: Dict[str, int] = defaultdict(int)
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.events: List[dict] = []
        self._seen_trades = 0
        self._last_exit_bar: Dict[str, int] = {}
        self._managed_snapshot: Dict[str, object] = {}
        self.bars_in_market = 0

    # ---------------------------------------------------------------- events --
    def emit(self, kind: str, **payload) -> None:
        event = {"kind": kind, **payload}
        self.events.append(event)
        if self.on_event:
            self.on_event(kind, event)

    # ------------------------------------------------------------- strategies --
    def add_strategy(self, strategy: Strategy) -> None:
        self.strategies.append(strategy)

    # ------------------------------------------------------------------ main --
    def on_candle(self, candle: Candle) -> None:
        symbol = candle.symbol
        history = self.history[symbol]
        history.append(candle)
        if len(history) > self.config.max_history_bars:
            del history[0]
        self.bar_index[symbol] += 1

        # 1) broker: mark to market + protective stops ----------------------- #
        self.broker.on_candle(candle)

        # 2) scaling bot: manage the open position --------------------------- #
        if self.scaling is not None:
            self._managed_snapshot.update(self.scaling.managed)
            for order in self.scaling.on_candle(candle, history):
                fill = self.broker.submit(order)
                if fill is not None and not order.reduce_only:
                    self.scaling.on_scale_in_filled(symbol, fill.price, fill.qty)

        # 3) strategies: look for new entries --------------------------------- #
        self._evaluate_entries(candle, history)

        # 4) bookkeeping ------------------------------------------------------ #
        if self.broker.positions():
            self.bars_in_market += 1
        self._absorb_new_trades()
        if self.config.record_equity:
            self._snapshot(candle.time)

    # --------------------------------------------------------------- entries --
    def _evaluate_entries(self, candle: Candle, history: List[Candle]) -> None:
        symbol = candle.symbol
        positions = self.broker.positions()
        position = positions.get(symbol)
        equity = self.broker.equity()

        for strategy in self.strategies:
            if len(history) < strategy.warmup:
                continue
            signal = strategy.on_candle(history)
            if signal is None:
                continue

            if position is not None and position.is_open:
                if self.config.exit_on_flip and position.side is not signal.side:
                    self.emit("exit_flip", symbol=symbol, price=candle.close, reason=signal.reason)
                    self.close_position(symbol, "signal_flip")
                    position = None
                else:
                    continue

            # cooldown after an exit
            last_exit = self._last_exit_bar.get(symbol)
            if (
                self.config.cooldown_bars
                and last_exit is not None
                and self.bar_index[symbol] - last_exit < self.config.cooldown_bars
            ):
                continue

            if self.risk is not None:
                allowed, reason = self.risk.can_open(equity, positions, candle.time)
                if not allowed:
                    self.emit("risk_block", symbol=symbol, reason=reason)
                    continue

            entry = candle.close
            stop = None
            if signal.stop_distance:
                stop = entry - signal.side.sign * signal.stop_distance

            # When the scaling bot plans several tranches, split the risk budget
            # so the *total* position still risks risk_per_trade_pct.
            tranches = 1
            if self.scaling is not None and self.scaling.config.tranches > 1:
                tranches = self.scaling.config.tranches
            if self.risk is not None:
                qty = self.risk.position_size(
                    symbol, equity, entry, stop, signal.strength, tranches=tranches
                )
            else:  # no risk manager -> fall back to one standard lot
                qty = 100_000.0
            if qty <= 0:
                self.emit("size_zero", symbol=symbol, reason=signal.reason)
                continue

            # trim to whatever exposure budget is left, then re-check
            if self.risk is not None:
                open_notional = sum(p.notional for p in positions.values())
                headroom = self.risk.max_qty_for_exposure(symbol, equity, open_notional, entry)
                if headroom <= 0:
                    self.emit("risk_block", symbol=symbol, reason="exposure cap reached")
                    continue
                qty = min(qty, headroom)

            order = Order(
                symbol=symbol,
                side=signal.side,
                qty=qty,
                order_type=OrderType.MARKET,
                stop_loss=stop,
                take_profit=signal.take_profit,
                tag=f"entry:{strategy.name}",
            )
            fill = self.broker.submit(order)
            if fill is not None:
                self.emit(
                    "entry",
                    symbol=symbol,
                    side=signal.side.value,
                    qty=fill.qty,
                    price=fill.price,
                    stop=stop,
                    reason=signal.reason,
                    strategy=strategy.name,
                )
                position = self.broker.positions().get(symbol)

    # ----------------------------------------------------------------- exits --
    def close_position(self, symbol: str, reason: str = "manual") -> Optional[Trade]:
        position = self.broker.positions().get(symbol)
        if position is None or not position.is_open:
            return None
        before = len(self.broker.closed_trades())
        self.broker.submit(
            Order(
                symbol=symbol,
                side=position.side.opposite,
                qty=position.qty,
                order_type=OrderType.MARKET,
                reduce_only=True,
                tag=reason,
            )
        )
        self._last_exit_bar[symbol] = self.bar_index[symbol]
        trades = self.broker.closed_trades()
        return trades[-1] if len(trades) > before else None

    # ------------------------------------------------------------ bookkeeping --
    def _absorb_new_trades(self) -> None:
        trades = self.broker.closed_trades()
        while self._seen_trades < len(trades):
            trade = trades[self._seen_trades]
            self._seen_trades += 1
            managed = self._managed_snapshot.get(trade.symbol)
            if managed is not None and trade.max_r == 0.0:
                trade.max_r = managed.max_progress_r
            # only forget the position once it is fully closed, so partial
            # scale-outs keep reporting the running max-R for the trade
            if not self.broker.positions().get(trade.symbol):
                self._managed_snapshot.pop(trade.symbol, None)
            if self.risk is not None:
                self.risk.register_equity(self.broker.equity(), trade.exit_time)
            if self.storage is not None:
                self.storage.record_trade(trade)
            self.emit(
                "trade_closed",
                symbol=trade.symbol,
                side=trade.side.value,
                qty=trade.qty,
                pnl=trade.net_pnl,
                reason=trade.exit_reason,
                max_r=trade.max_r,
            )

    def _snapshot(self, when: datetime) -> None:
        equity = self.broker.equity()
        self.equity_curve.append((when, equity))
        if self.risk is not None:
            self.risk.register_equity(equity, when)
        if self.storage is not None:
            self.storage.record_equity(
                when, equity, self.broker.cash(), self.broker.unrealized_pnl()
            )

    # --------------------------------------------------------------- results --
    def equity(self) -> float:
        return self.broker.equity()

    def trades(self) -> List[Trade]:
        return self.broker.closed_trades()

    def open_positions(self) -> Dict[str, Position]:
        return self.broker.positions()
