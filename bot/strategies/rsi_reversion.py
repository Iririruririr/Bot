"""RSI mean reversion - buys dips and sells rips in ranging FX markets.

Enters when RSI leaves an oversold/overbought extreme and exits when it returns
to the mid line (50) or when a time-based stop triggers in the engine.
"""
from __future__ import annotations

from typing import List, Optional

from bot.core.models import Candle, Signal, Side
from bot.strategies.base import Strategy, register
from bot.strategies.indicators import rsi


@register
class RSIMeanReversion(Strategy):
    name = "rsi_reversion"
    warmup = 30

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_level: float = 50.0,
        atr_period: int = 14,
        atr_mult: float = 1.5,
        **params,
    ):
        super().__init__(
            period=period,
            oversold=oversold,
            overbought=overbought,
            exit_level=exit_level,
            atr_period=atr_period,
            atr_mult=atr_mult,
            **params,
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_level = exit_level
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def on_candle(self, history: List[Candle]) -> Optional[Signal]:
        closes = self._closes(history)
        if len(closes) < self.period + 3:
            return None
        values = rsi(closes, self.period)
        if len(values) < 2:
            return None

        now, prev = values[-1], values[-2]
        atr_value = self._atr(history, self.atr_period) or (closes[-1] * 0.001)
        stop_distance = self.atr_mult * atr_value

        if prev <= self.oversold < now:
            strength = (self.oversold - prev) / max(self.oversold, 1e-9)
            return Signal(
                symbol=history[-1].symbol,
                side=Side.BUY,
                strength=max(0.25, min(strength, 1.0)),
                stop_distance=stop_distance,
                reason=f"RSI left oversold ({now:.1f})",
            )
        if prev >= self.overbought > now:
            strength = (prev - self.overbought) / max(100.0 - self.overbought, 1e-9)
            return Signal(
                symbol=history[-1].symbol,
                side=Side.SELL,
                strength=max(0.25, min(strength, 1.0)),
                stop_distance=stop_distance,
                reason=f"RSI left overbought ({now:.1f})",
            )
        return None
