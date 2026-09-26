"""Shared test helpers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.core.models import Candle

START = datetime(2024, 1, 2, tzinfo=timezone.utc)


def make_candle(
    symbol: str,
    close: float,
    index: int = 0,
    open_price: float | None = None,
    high: float | None = None,
    low: float | None = None,
    wick: float = 0.0002,
) -> Candle:
    """Build a plausible candle around a close price."""
    open_price = close if open_price is None else open_price
    high = max(open_price, close) + wick if high is None else high
    low = min(open_price, close) - wick if low is None else low
    return Candle(
        symbol=symbol,
        time=START + timedelta(hours=index),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
    )


def candle_series(symbol: str, closes, wick: float = 0.0002, start_index: int = 0):
    return [
        make_candle(symbol, close, index=start_index + i, wick=wick)
        for i, close in enumerate(closes)
    ]


def walk(series, callback):
    """Feed a series of candles to ``callback`` one at a time."""
    for candle in series:
        callback(candle)


@pytest.fixture
def eurusd():
    return "EUR/USD"


@pytest.fixture
def usdjpy():
    return "USD/JPY"
