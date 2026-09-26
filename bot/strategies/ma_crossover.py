"""Moving-average crossover - the classic trend-following FX strategy.

Long when the fast MA crosses above the slow MA, short when it crosses below.
The protective stop is derived from ATR so it adapts to each pair's volatility.
"""
from __future__ import annotations

from typing import List, Optional

from bot.core.models import Candle, Signal, Side
from bot.strategies.base import Strategy, register
from bot.strategies.indicators import sma

# Two moving averages computed over a flat stretch can differ by a few ULPs, so
# the crossover comparison needs a tolerance or a real cross gets missed.
_EPS = 1e-12


@register
class MovingAverageCrossover(Strategy):
    name = "ma_crossover"
    warmup = 60

    def __init__(self, fast: int = 20, slow: int = 50, atr_period: int = 14, atr_mult: float = 2.0, **params):
        super().__init__(fast=fast, slow=slow, atr_period=atr_period, atr_mult=atr_mult, **params)
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def on_candle(self, history: List[Candle]) -> Optional[Signal]:
        closes = self._closes(history)
        if len(closes) < self.slow + 2:
            return None

        fast = sma(closes, self.fast)
        slow = sma(closes, self.slow)
        if len(fast) < 2 or len(slow) < 2:
            return None

        # align the tail of both series and look at the last two points
        f_now, f_prev = fast[-1], fast[-2]
        s_now, s_prev = slow[-1], slow[-2]

        atr_value = self._atr(history, self.atr_period) or (closes[-1] * 0.001)
        stop_distance = self.atr_mult * atr_value

        crossed_up = f_prev <= s_prev + _EPS and f_now > s_now + _EPS
        crossed_down = f_prev >= s_prev - _EPS and f_now < s_now - _EPS

        if crossed_up:
            return Signal(
                symbol=history[-1].symbol,
                side=Side.BUY,
                stop_distance=stop_distance,
                reason=f"SMA{self.fast} crossed above SMA{self.slow}",
            )
        if crossed_down:
            return Signal(
                symbol=history[-1].symbol,
                side=Side.SELL,
                stop_distance=stop_distance,
                reason=f"SMA{self.fast} crossed below SMA{self.slow}",
            )
        return None
