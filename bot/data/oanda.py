"""Live FX data and execution via OANDA's v20 REST API.

OANDA is used because it offers free **practice** accounts, real streaming
prices for ~120 FX pairs and a well documented API.  Credentials come from the
environment (never from the repo):

    export OANDA_TOKEN=xxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx
    export OANDA_ACCOUNT=101-001-1234567-001
    export OANDA_HOST=api-fxpractice.oanda.com     # or api-fxtrade.oanda.com

Everything is stdlib-only (``urllib``), so there is no extra dependency for
people who only want the simulated/backtest paths.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from bot.core.models import Candle, Fill, Order, Position, Quote, Side, Trade, spec_for
from bot.data.feed import DataFeed

DEFAULT_HOST = "api-fxpractice.oanda.com"

# our timeframe label -> OANDA granularity
GRANULARITY = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "4h": "H4",
    "1d": "D",
}


class OandaError(RuntimeError):
    pass


class OandaClient:
    """Thin REST client.  ``transport`` is injectable for tests."""

    def __init__(
        self,
        token: Optional[str] = None,
        account: Optional[str] = None,
        host: Optional[str] = None,
        transport: Optional[Callable[[str, str, Optional[dict], Optional[dict]], dict]] = None,
    ):
        self.token = token or os.environ.get("OANDA_TOKEN", "")
        self.account = account or os.environ.get("OANDA_ACCOUNT", "")
        self.host = host or os.environ.get("OANDA_HOST", DEFAULT_HOST)
        self._transport = transport or self._default_transport
        if not self.token or not self.account:
            raise OandaError(
                "OANDA credentials missing - set OANDA_TOKEN and OANDA_ACCOUNT "
                "(practice accounts are free at oanda.com)"
            )

    @property
    def base_url(self) -> str:
        return f"https://{self.host}/v3"

    # ------------------------------------------------------------ transport --
    def _default_transport(
        self, method: str, url: str, body: Optional[dict], headers: Optional[dict]
    ) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = response.read().decode()
                return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise OandaError(f"OANDA HTTP {exc.code}: {exc.read().decode()[:300]}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise OandaError(f"OANDA connection failed: {exc.reason}") from exc

    def _request(self, method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return self._transport(method, url, body, None)

    # --------------------------------------------------------------- pricing --
    def price(self, instrument: str) -> Quote:
        data = self._request(
            "GET",
            f"/accounts/{self.account}/pricing",
            params={"instruments": instrument},
        )
        prices = data.get("prices") or []
        if not prices:
            raise OandaError(f"no price returned for {instrument}")
        entry = prices[0]
        bids = [b["price"] for b in entry.get("bids", [])]
        asks = [a["price"] for a in entry.get("asks", [])]
        return Quote(
            symbol=instrument.replace("_", "/"),
            time=datetime.now(timezone.utc),
            bid=float(bids[0]) if bids else 0.0,
            ask=float(asks[0]) if asks else 0.0,
        )

    def candles(self, instrument: str, count: int = 500, granularity: str = "H1") -> List[Candle]:
        data = self._request(
            "GET",
            f"/instruments/{instrument}/candles",
            params={"count": count, "granularity": granularity, "price": "M"},
        )
        bars: List[Candle] = []
        for entry in data.get("candles", []):
            if not entry.get("complete", True):
                continue
            mid = entry["mid"]
            bars.append(
                Candle(
                    symbol=instrument.replace("_", "/"),
                    time=datetime.fromisoformat(entry["time"].replace("Z", "+00:00")),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=float(entry.get("volume", 0.0)),
                )
            )
        return bars

    # --------------------------------------------------------------- trading --
    def market_order(
        self,
        instrument: str,
        units: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> dict:
        order: Dict[str, object] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if stop_loss is not None:
            order["stopLossOnFill"] = {"price": f"{stop_loss:.5f}", "timeInForce": "GTC"}
        if take_profit is not None:
            order["takeProfitOnFill"] = {"price": f"{take_profit:.5f}", "timeInForce": "GTC"}
        return self._request("POST", f"/accounts/{self.account}/orders", {"order": order})

    def close_position(self, instrument: str) -> dict:
        """Close whichever side of the position is open."""
        data = self._request("GET", f"/accounts/{self.account}/positions")
        for entry in data.get("positions", []):
            if entry["instrument"] != instrument:
                continue
            if float(entry.get("long", {}).get("units") or 0):
                return self._request(
                    "PUT", f"/accounts/{self.account}/positions/{instrument}/close", {"longUnits": "ALL"}
                )
            if float(entry.get("short", {}).get("units") or 0):
                return self._request(
                    "PUT", f"/accounts/{self.account}/positions/{instrument}/close", {"shortUnits": "ALL"}
                )
        return {}

    def open_positions(self) -> List[dict]:
        data = self._request("GET", f"/accounts/{self.account}/openPositions")
        return data.get("positions", [])

    def account_summary(self) -> dict:
        return self._request("GET", f"/accounts/{self.account}/summary").get("account", {})


class OandaFeed(DataFeed):
    """Live candle feed backed by OANDA."""

    def __init__(self, client: OandaClient, instrument: str, timeframe: str = "1h", count: int = 500):
        self.client = client
        self.symbol = instrument.replace("_", "/")
        self.instrument = instrument
        self.timeframe = timeframe
        self.count = count
        self._bars: Optional[List[Candle]] = None

    def generate(self) -> List[Candle]:
        if self._bars is None:
            self._bars = self.client.candles(
                self.instrument, count=self.count, granularity=GRANULARITY.get(self.timeframe, "H1")
            )
        return self._bars

    def refresh(self) -> List[Candle]:
        self._bars = None
        return self.generate()

    def __iter__(self):
        return iter(self.generate())

    def __len__(self) -> int:
        return len(self.generate())

    def quote(self) -> Quote:
        return self.client.price(self.instrument)


class OandaBroker:
    """Live execution adapter (used by ``python -m bot live``).

    Positions are netted per instrument exactly like the paper broker, so the
    same engine drives both.
    """

    def __init__(self, client: OandaClient, account_ccy: str = "USD", spread_pips: float = 0.0):
        self.client = client
        self.config_account_ccy = account_ccy
        self.spread_pips = spread_pips
        self._specs: Dict[str, object] = {}

    def spec(self, symbol: str):
        key = symbol.replace("/", "_")
        if key not in self._specs:
            self._specs[key] = spec_for(key)
        return self._specs[key]

    def quote(self, symbol: str) -> Quote:
        return self.client.price(symbol.replace("/", "_"))

    def update_price(self, symbol: str, bid: float, ask: float, when: Optional[datetime] = None) -> None:
        self._last = Quote(symbol=symbol, time=when or datetime.now(timezone.utc), bid=bid, ask=ask)

    def on_candle(self, candle: Candle) -> List[Trade]:
        # OANDA holds the stops server-side; nothing to simulate here
        return []

    def submit(self, order: Order) -> Optional[Fill]:
        instrument = order.symbol.replace("/", "_")
        units = order.qty * (1 if order.side is Side.BUY else -1)
        try:
            result = self.client.market_order(
                instrument,
                units,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
            )
        except OandaError:
            return None
        fill = (result.get("orderFillTransaction") or {})
        price = float(fill.get("price") or 0.0)
        if price <= 0:
            return None
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            qty=float(fill.get("units") or order.qty),
            price=price,
            time=datetime.now(timezone.utc),
            commission=float(fill.get("commission") or 0.0),
            tag=order.tag,
        )

    def positions(self) -> Dict[str, Position]:
        out: Dict[str, Position] = {}
        for entry in self.client.open_positions():
            instrument = entry["instrument"]
            symbol = instrument.replace("_", "/")
            long_units = float(entry.get("long", {}).get("units") or 0.0)
            short_units = float(entry.get("short", {}).get("units") or 0.0)
            if long_units:
                out[symbol] = Position(
                    symbol=symbol,
                    side=Side.BUY,
                    qty=long_units,
                    avg_price=float(entry["long"]["averagePrice"]),
                    initial_qty=long_units,
                )
            elif short_units:
                out[symbol] = Position(
                    symbol=symbol,
                    side=Side.SELL,
                    qty=short_units,
                    avg_price=float(entry["short"]["averagePrice"]),
                    initial_qty=short_units,
                )
        return out

    def cash(self) -> float:
        return float(self.client.account_summary().get("balance", 0.0))

    def equity(self) -> float:
        return float(self.client.account_summary().get("NAV", 0.0))

    def unrealized_pnl(self) -> float:
        return float(self.client.account_summary().get("unrealizedPL", 0.0))

    def open_notional(self) -> float:
        return sum(p.notional for p in self.positions().values())

    def closed_trades(self) -> List[Trade]:
        return []  # the live broker does not keep a local trade log


__all__ = ["OandaClient", "OandaFeed", "OandaBroker", "OandaError", "GRANULARITY"]
