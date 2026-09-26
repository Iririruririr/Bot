"""Pure-python technical indicators.

Deliberately dependency free (and O(n) per call on a rolling window) so the
strategies can be reused in the backtester, the live loop and unit tests
without pulling in pandas.
"""
from __future__ import annotations

from typing import List, Sequence


def sma(values: Sequence[float], period: int) -> List[float]:
    """Simple moving average; leading entries are ``None``-free (list is shorter)."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: List[float] = []
    window_sum = 0.0
    for i, v in enumerate(values):
        window_sum += v
        if i >= period:
            window_sum -= values[i - period]
        if i >= period - 1:
            out.append(window_sum / period)
    return out


def ema(values: Sequence[float], period: int) -> List[float]:
    """Exponential moving average (seeded with the first ``period`` SMA)."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return []
    k = 2.0 / (period + 1.0)
    out = [sum(values[:period]) / period]
    for v in values[period:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out


def rsi(values: Sequence[float], period: int = 14) -> List[float]:
    """Wilder's RSI."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) <= period:
        return []
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out: List[float] = []
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        if avg_loss == 0:
            out.append(100.0)
        else:
            rs = avg_gain / avg_loss
            out.append(100.0 - 100.0 / (1.0 + rs))
    return out


def true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> List[float]:
    out = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        out.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - prev_close),
                abs(lows[i] - prev_close),
            )
        )
    return out


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> List[float]:
    """Average true range (Wilder smoothing)."""
    if period <= 0:
        raise ValueError("period must be positive")
    tr = true_range(highs, lows, closes)
    if len(tr) < period:
        return []
    out = [sum(tr[:period]) / period]
    for value in tr[period:]:
        out.append((out[-1] * (period - 1) + value) / period)
    return out


def rolling_max(values: Sequence[float], period: int) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        if i < period - 1:
            continue
        out.append(max(values[i - period + 1 : i + 1]))
    return out


def rolling_min(values: Sequence[float], period: int) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        if i < period - 1:
            continue
        out.append(min(values[i - period + 1 : i + 1]))
    return out


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def zscore(values: Sequence[float], period: int) -> List[float]:
    """Rolling z-score of the last value versus the previous ``period`` values."""
    out: List[float] = []
    for i in range(len(values)):
        if i < period:
            continue
        window = values[i - period : i]
        mu = sum(window) / period
        sd = stdev(window)
        out.append(0.0 if sd == 0 else (values[i] - mu) / sd)
    return out
