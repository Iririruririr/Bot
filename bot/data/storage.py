"""SQLite persistence for trades, equity snapshots and events.

Everything lives in one file (``data/bot.db`` by default) so a run can be
inspected later with the ``report`` command, or opened directly in any SQLite
client.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from bot.core.models import Trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL,
    pnl REAL NOT NULL,
    commission REAL NOT NULL,
    exit_reason TEXT,
    tags TEXT,
    max_r REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    unrealized REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    time TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, time)
);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
CREATE INDEX IF NOT EXISTS idx_equity_time ON equity(time);
"""


class SQLiteStore:
    def __init__(self, path: str = "data/bot.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ write --
    def record_trade(self, trade: Trade) -> None:
        self.conn.execute(
            "INSERT INTO trades (symbol, side, qty, entry_price, exit_price, entry_time,"
            " exit_time, pnl, commission, exit_reason, tags, max_r)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade.symbol,
                trade.side.value,
                trade.qty,
                trade.entry_price,
                trade.exit_price,
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat(),
                trade.pnl,
                trade.commission,
                trade.exit_reason,
                json.dumps(trade.tags),
                trade.max_r,
            ),
        )
        self.conn.commit()

    def record_equity(self, when: datetime, equity: float, cash: float, unrealized: float = 0.0) -> None:
        self.conn.execute(
            "INSERT INTO equity (time, equity, cash, unrealized) VALUES (?,?,?,?)",
            (when.isoformat(), equity, cash, unrealized),
        )
        self.conn.commit()

    def record_event(self, when: datetime, kind: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO events (time, kind, payload) VALUES (?,?,?)",
            (when.isoformat(), kind, json.dumps(payload, default=str)),
        )
        self.conn.commit()

    def record_candles(self, candles: Iterable) -> None:
        rows = [
            (c.symbol, c.time.isoformat(), c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO candles (symbol, time, open, high, low, close, volume)"
            " VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    # ------------------------------------------------------------------- read --
    def trades(self, limit: Optional[int] = None) -> List[Trade]:
        query = "SELECT * FROM trades ORDER BY exit_time"
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = self.conn.execute(query).fetchall()
        out: List[Trade] = []
        # columns: id, symbol, side, qty, entry, exit, entry_time, exit_time,
        #          pnl, commission, exit_reason, tags, max_r
        for row in rows:
            out.append(
                Trade(
                    symbol=row[1],
                    side=_side_from_text(row[2]),
                    qty=row[3],
                    entry_price=row[4],
                    exit_price=row[5],
                    entry_time=datetime.fromisoformat(row[6]),
                    exit_time=datetime.fromisoformat(row[7]),
                    pnl=row[8],
                    commission=row[9],
                    exit_reason=row[10] or "",
                    tags=json.loads(row[11] or "[]"),
                    max_r=row[12] or 0.0,
                )
            )
        return out

    def equity_curve(self) -> List[Tuple[datetime, float]]:
        rows = self.conn.execute("SELECT time, equity FROM equity ORDER BY time").fetchall()
        return [(datetime.fromisoformat(r[0]), r[1]) for r in rows]

    def events(self, limit: int = 200) -> List[dict]:
        rows = self.conn.execute(
            "SELECT time, kind, payload FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"time": r[0], "kind": r[1], "payload": json.loads(r[2] or "{}")} for r in rows
        ]

    def candle_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]

    def close(self) -> None:
        self.conn.close()


def _side_from_text(value: str):
    from bot.core.models import Side

    return Side.BUY if value.upper() == "BUY" else Side.SELL
