"""Strategy interface + the strategy registry.

A strategy is a pure function of bar history: it sees the candles up to and
including the current one and returns a :class:`~bot.core.models.Signal` when it
wants to be in the market.  It never touches the broker, so it is trivially
testable and reusable in backtests, paper runs and live runs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from bot.core.models import Candle, Signal
from bot.strategies.indicators import atr


class Strategy(ABC):
    """Base class for every strategy."""

    name: str = "base"
    warmup: int = 30  # bars of history needed before the strategy may signal

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def on_candle(self, history: List[Candle]) -> Optional[Signal]:
        """Called once per closed bar.  ``history[-1]`` is the current bar."""

    # ------------------------------------------------------------- helpers --
    @staticmethod
    def _atr(history: List[Candle], period: int = 14) -> Optional[float]:
        """Latest ATR, or ``None`` when there is not enough history."""
        if len(history) < period + 1:
            return None
        values = atr(
            [c.high for c in history],
            [c.low for c in history],
            [c.close for c in history],
            period,
        )
        return values[-1] if values else None

    @staticmethod
    def _closes(history: List[Candle]) -> List[float]:
        return [c.close for c in history]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.name} {self.params}>"


REGISTRY: Dict[str, Type[Strategy]] = {}


def register(cls: Type[Strategy]) -> Type[Strategy]:
    """Class decorator that adds a strategy to the registry."""
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def available() -> List[str]:
    return sorted(REGISTRY)
