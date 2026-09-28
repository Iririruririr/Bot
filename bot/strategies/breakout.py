"""Donchian-style breakout - enters on new N-bar highs/lows (turtle trading).

FX trends hard once a range breaks, so this strategy keeps a wide ATR stop and
hands the position straight to the scaling bot to scale into the follow-through.
"""
from __future__ import annotations

from typing import List, Optional

from bot.core.models import Candle, Signal, Side
from bot.strategies.base import Strategy, register
from bot.strategies.indicators import rolling_max, rolling_min


@register
class Breakout(Strategy):
    name = "breakout"
    warmup = 30

    def __init__(self, lookback: int = 20, atr_period: int = 14, atr_mult: float = 2.5, **params):
        super().__init__(lookback=lookback, atr_period=atr_period, atr_mult=atr_mult, **params)
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def on_candle(self, history: List[Candle]) -> Optional[Signal]:
        closes = self._closes(history)
        highs = [c.high for c in history]
        lows = [c.low for c in history]
        if len(closes) < self.lookback + 2:
            return None

        upper = rolling_max(highs[:-1], self.lookback)  # exclude the current bar
        lower = rolling_min(lows[:-1], self.lookback)
        if not upper or not lower:
            return None

        price = closes[-1]
        atr_value = self._atr(history, self.atr_period) or (price * 0.001)
        stop_distance = self.atr_mult * atr_value

        if price > upper[-1]:
            return Signal(
                symbol=history[-1].symbol,
                side=Side.BUY,
                stop_distance=stop_distance,
                reason=f"break above {self.lookback}-bar high",
            )
        if price < lower[-1]:
            return Signal(
                symbol=history[-1].symbol,
                side=Side.SELL,
                stop_distance=stop_distance,
                reason=f"break below {self.lookback}-bar low",
            )
        return None
