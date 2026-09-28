"""Market data feeds.

Three sources, one interface:

* :class:`SimulatedFeed` - deterministic geometric Brownian motion with a slow
  trend cycle.  This is the default: the bots run and backtest offline with no
  API keys and no internet.
* :class:`CsvFeed` - replays a CSV of historical bars (``time,open,high,low,close,volume``).
* :class:`OandaFeed` - live prices and candles from OANDA's REST API (practice
  or live account), used by the paper/live runner.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from bot.core.models import Candle, Quote, spec_for

# seconds per bar, used to space out simulated bars realistically
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _parse_time(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class DataFeed:
    """Common interface - every feed is an iterable of candles."""

    symbol: str = ""
    timeframe: str = "1h"

    def __iter__(self) -> Iterator[Candle]:  # pragma: no cover - interface
        raise NotImplementedError

    def quote(self) -> Quote:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class SimConfig:
    symbol: str = "EUR/USD"
    start_price: float = 1.0800
    bars: int = 2000
    timeframe: str = "1h"
    vol_pips: float = 12.0        # per-bar standard deviation
    drift_pips: float = 0.0       # per-bar drift
    cycle_bars: int = 500         # length of the slow trend cycle
    cycle_amp_pips: float = 1.5   # extra drift at the peak of the cycle
    wick_pips: float = 6.0        # average wick size
    seed: int = 7
    start: Optional[datetime] = None


class SimulatedFeed(DataFeed):
    """Deterministic synthetic FX series - good enough to develop and test against.

    Price follows a log-normal random walk with a sinusoidal drift component so
    the feed contains both trending and ranging regimes (otherwise trend and
    mean-reversion strategies cannot both be evaluated).
    """

    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        self.symbol = self.config.symbol
        self.timeframe = self.config.timeframe
        self._rng = random.Random(self.config.seed)
        self._bars: Optional[List[Candle]] = None

    # ------------------------------------------------------------------ build --
    def generate(self) -> List[Candle]:
        if self._bars is not None:
            return self._bars
        cfg = self.config
        spec = spec_for(cfg.symbol)
        pip = spec.pip_size
        step = timedelta(seconds=TIMEFRAME_SECONDS.get(cfg.timeframe, 3600))
        start = cfg.start or datetime(2024, 1, 2, tzinfo=timezone.utc)

        price = cfg.start_price
        bars: List[Candle] = []
        for i in range(cfg.bars):
            # slow sinusoidal drift gives the series regime changes
            cycle = math.sin(2 * math.pi * i / max(cfg.cycle_bars, 1))
            drift = (cfg.drift_pips + cfg.cycle_amp_pips * cycle) * pip / price
            shock = self._rng.gauss(0.0, cfg.vol_pips * pip / price)
            ret = drift + shock

            open_price = price
            close_price = open_price * math.exp(ret)
            wick = abs(self._rng.gauss(0.0, cfg.wick_pips * pip))
            high = max(open_price, close_price) + wick
            low = min(open_price, close_price) - wick
            volume = max(1.0, self._rng.gauss(1000.0, 250.0))

            bars.append(
                Candle(
                    symbol=cfg.symbol,
                    time=start + i * step,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=volume,
                )
            )
            price = close_price
        self._bars = bars
        return bars

    def __iter__(self) -> Iterator[Candle]:
        return iter(self.generate())

    def __len__(self) -> int:
        return len(self.generate())

    def quote(self) -> Quote:
        bars = self.generate()
        last = bars[-1]
        spec = spec_for(self.symbol)
        half = spec.pip_size / 2.0
        return Quote(symbol=self.symbol, time=last.time, bid=last.close - half, ask=last.close + half)


class CsvFeed(DataFeed):
    """Replays historical bars from a CSV file."""

    def __init__(self, path: str, symbol: Optional[str] = None, timeframe: str = "1h"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such data file: {self.path}")
        self._bars: Optional[List[Candle]] = None
        header_symbol = symbol
        if header_symbol is None:
            with self.path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                first = next(reader, None)
                header_symbol = (first or {}).get("symbol") or self.path.stem
        self.symbol = header_symbol
        self.timeframe = timeframe

    def generate(self) -> List[Candle]:
        if self._bars is not None:
            return self._bars
        bars: List[Candle] = []
        with self.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                bars.append(
                    Candle(
                        symbol=row.get("symbol") or self.symbol,
                        time=_parse_time(row["time"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row.get("volume") or 0.0),
                    )
                )
        bars.sort(key=lambda c: c.time)
        self._bars = bars
        return bars

    def __iter__(self) -> Iterator[Candle]:
        return iter(self.generate())

    def __len__(self) -> int:
        return len(self.generate())

    def quote(self) -> Quote:
        bars = self.generate()
        last = bars[-1]
        spec = spec_for(self.symbol)
        half = spec.pip_size / 2.0
        return Quote(symbol=self.symbol, time=last.time, bid=last.close - half, ask=last.close + half)


# --------------------------------------------------------------------------- #
# Historical data generation helpers
# --------------------------------------------------------------------------- #


def generate_csv(
    path: str,
    symbol: str = "EUR/USD",
    bars: int = 5000,
    timeframe: str = "1h",
    seed: int = 11,
    start_price: Optional[float] = None,
) -> str:
    """Write a synthetic historical CSV (handy for offline backtests)."""
    if start_price is None:
        start_price = {"USD/JPY": 150.0, "GBP/USD": 1.26, "EUR/USD": 1.08}.get(symbol.upper(), 1.0)
    feed = SimulatedFeed(
        SimConfig(symbol=symbol, start_price=start_price, bars=bars, timeframe=timeframe, seed=seed)
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "symbol", "open", "high", "low", "close", "volume"])
        for candle in feed:
            writer.writerow(
                [
                    candle.time.isoformat(),
                    candle.symbol,
                    f"{candle.open:.5f}",
                    f"{candle.high:.5f}",
                    f"{candle.low:.5f}",
                    f"{candle.close:.5f}",
                    f"{candle.volume:.0f}",
                ]
            )
    return str(target)
