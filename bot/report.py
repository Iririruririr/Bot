"""Terminal reporting - tables, equity sparkline and trade lists."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

from bot.backtest.stats import Stats
from bot.core.models import Trade

BAR_CHARS = " ▁▂▃▄▅▆▇█"


def _fmt_money(value: float, currency: str = "") -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{currency}{abs(value):,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def sparkline(values: Sequence[float], width: int = 60) -> str:
    """Tiny ASCII chart of a series."""
    if not values:
        return "(no data)"
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return BAR_CHARS[4] * len(values)
    span = hi - lo
    return "".join(BAR_CHARS[int((v - lo) / span * (len(BAR_CHARS) - 1))] for v in values)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "") -> str:
    """Render a simple aligned table."""
    if not rows:
        return f"{indent}(none)"
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = []
    header = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(indent + header)
    lines.append(indent + "  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(indent + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def summary_report(
    stats: Stats,
    title: str = "Backtest summary",
    currency: str = "$",
    extra_rows: Optional[Sequence[Tuple[str, str]]] = None,
) -> str:
    lines = [f"\n=== {title} ==="]
    rows = [
        ("Net P&L", _fmt_money(stats.net_pnl, currency)),
        ("Return", _fmt_pct(stats.total_return_pct)),
        ("Final equity", _fmt_money(stats.final_equity, currency)),
        ("Max drawdown", f"{stats.max_drawdown_pct:.2f}%  ({_fmt_money(stats.max_drawdown_amount, currency)})"),
        ("Sharpe", f"{stats.sharpe:.2f}"),
        ("Sortino", f"{stats.sortino:.2f}"),
        ("Profit factor", f"{stats.profit_factor:.2f}"),
        ("Trades", f"{stats.trades}"),
        ("Win rate", f"{stats.win_rate:.1f}%  ({stats.wins}W / {stats.losses}L)"),
        ("Expectancy", f"{_fmt_money(stats.expectancy, currency)} / trade"),
        ("Expectancy (R)", f"{stats.expectancy_r:+.2f}R" if stats.expectancy_r else "n/a"),
        ("Avg win / loss", f"{_fmt_money(stats.avg_win, currency)} / {_fmt_money(stats.avg_loss, currency)}"),
        ("Largest win / loss", f"{_fmt_money(stats.largest_win, currency)} / {_fmt_money(stats.largest_loss, currency)}"),
        ("Commission", _fmt_money(stats.total_commission, currency)),
        ("Exposure", f"{stats.exposure_pct:.1f}% of bars"),
    ]
    if extra_rows:
        rows.extend(extra_rows)
    lines.append(table(["Metric", "Value"], rows))
    return "\n".join(lines)


def equity_report(equity_curve: Sequence[Tuple[datetime, float]], width: int = 64) -> str:
    if not equity_curve:
        return "(no equity data)"
    values = [v for _, v in equity_curve]
    start, end = equity_curve[0], equity_curve[-1]
    return (
        f"\nEquity curve ({start[0]:%Y-%m-%d} -> {end[0]:%Y-%m-%d})\n"
        f"{sparkline(values, width)}\n"
        f"start {_fmt_money(start[1])}   end {_fmt_money(end[1])}   "
        f"change {_fmt_pct((end[1] - start[1]) / start[1] * 100 if start[1] else 0)}"
    )


def trades_report(trades: Sequence[Trade], limit: int = 25, currency: str = "$") -> str:
    if not trades:
        return "\nNo closed trades."
    shown = trades[-limit:]
    rows = [
        (
            t.exit_time.strftime("%Y-%m-%d %H:%M"),
            t.symbol,
            t.side.value,
            f"{t.qty:,.0f}",
            f"{t.entry_price:.5f}",
            f"{t.exit_price:.5f}",
            _fmt_money(t.net_pnl, currency),
            f"{t.max_r:+.1f}R",
            t.exit_reason,
        )
        for t in shown
    ]
    header = "\n=== Recent trades ===" + (f" (last {len(shown)} of {len(trades)})" if len(trades) > len(shown) else "")
    return header + "\n" + table(
        ["Exit time", "Symbol", "Side", "Qty", "Entry", "Exit", "P&L", "MaxR", "Reason"], rows
    )


def scaling_report(scaling_stats: dict, events: Sequence[dict], limit: int = 12) -> str:
    """Report on what the scaling bot actually did."""
    lines = ["\n=== Scaling bot activity ==="]
    lines.append(
        table(
            ["Action", "Count"],
            [
                ("Scale-in adds", scaling_stats.get("adds", 0)),
                ("Partial exits", scaling_stats.get("scale_outs", 0)),
                ("Breakeven moves", scaling_stats.get("breakevens", 0)),
                ("Trail updates", scaling_stats.get("trail_updates", 0)),
                ("Adds skipped (caps)", scaling_stats.get("skipped_adds", 0)),
            ],
        )
    )
    interesting = [
        e for e in events if e.get("kind") in ("scale_in", "scale_out", "breakeven", "trail", "time_stop", "risk_rebalanced")
    ][-limit:]
    if interesting:
        rows = []
        for event in interesting:
            kind = event["kind"]
            if kind == "scale_in":
                detail = f"tranche {event.get('tranche')} +{event.get('qty'):,.0f} @ {event.get('price'):.5f}"
            elif kind == "scale_out":
                detail = f"{event.get('r')}R close {event.get('qty'):,.0f} @ {event.get('price'):.5f}"
            elif kind == "breakeven":
                detail = f"stop -> {event.get('stop'):.5f}"
            elif kind == "trail":
                detail = f"stop -> {event.get('stop'):.5f}"
            elif kind == "time_stop":
                detail = f"closed after {event.get('bars')} bars"
            else:
                detail = f"stop -> {event.get('stop'):.5f} on {event.get('qty'):,.0f}"
            rows.append((event.get("symbol", ""), kind, detail))
        lines.append("\n" + table(["Symbol", "Event", "Detail"], rows))
    return "\n".join(lines)


def comparison_report(results: Sequence[Tuple[str, Stats]]) -> str:
    """Side-by-side comparison of several backtests."""
    rows = []
    for name, stats in results:
        rows.append(
            (
                name,
                _fmt_money(stats.net_pnl),
                _fmt_pct(stats.total_return_pct),
                f"{stats.max_drawdown_pct:.1f}%",
                f"{stats.sharpe:.2f}",
                f"{stats.win_rate:.0f}%",
                str(stats.trades),
                f"{stats.expectancy_r:+.2f}R",
            )
        )
    return "\n=== Strategy comparison ===\n" + table(
        ["Run", "Net P&L", "Return", "MaxDD", "Sharpe", "Win%", "Trades", "Exp(R)"], rows
    )
