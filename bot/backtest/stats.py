"""Performance statistics for a completed backtest or paper session."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

from bot.core.models import Trade

# bars per calendar year, used to annualise the Sharpe ratio
BARS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 252,
}


@dataclass
class Stats:
    starting_equity: float
    final_equity: float
    trades: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0            # account currency per trade
    expectancy_r: float = 0.0          # average R multiple per trade
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_amount: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    exposure_pct: float = 0.0          # % of bars with an open position
    avg_bars_in_trade: float = 0.0
    total_commission: float = 0.0
    by_reason: Dict[str, Tuple[int, float]] = field(default_factory=dict)
    by_symbol: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    @property
    def cagr(self) -> float:
        return 0.0  # filled in by compute_stats when dates are available


def _max_drawdown(curve: Sequence[Tuple[datetime, float]]) -> Tuple[float, float]:
    peak = curve[0][1] if curve else 0.0
    max_pct, max_amount = 0.0, 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        amount = peak - equity
        max_amount = max(max_amount, amount)
        if peak > 0:
            max_pct = max(max_pct, amount / peak * 100.0)
    return max_pct, max_amount


def compute_stats(
    trades: Sequence[Trade],
    equity_curve: Sequence[Tuple[datetime, float]],
    starting_equity: float,
    timeframe: str = "1h",
    bars_in_market: Optional[int] = None,
    total_bars: Optional[int] = None,
) -> Stats:
    stats = Stats(starting_equity=starting_equity, final_equity=equity_curve[-1][1] if equity_curve else starting_equity)

    # drawdown is a property of the equity curve, so it applies even with no trades
    stats.max_drawdown_pct, stats.max_drawdown_amount = _max_drawdown(equity_curve)

    if not trades:
        stats.final_equity = starting_equity
        return stats

    pnls = [t.net_pnl for t in trades]
    stats.trades = len(pnls)
    stats.wins = sum(1 for p in pnls if p > 0)
    stats.losses = sum(1 for p in pnls if p < 0)
    stats.scratches = stats.trades - stats.wins - stats.losses
    stats.win_rate = stats.wins / stats.trades * 100.0
    stats.gross_profit = sum(p for p in pnls if p > 0)
    stats.gross_loss = -sum(p for p in pnls if p < 0)
    stats.net_pnl = sum(pnls)
    stats.total_commission = sum(t.commission for t in trades)
    stats.total_return_pct = (
        (stats.final_equity - starting_equity) / starting_equity * 100.0 if starting_equity else 0.0
    )
    stats.profit_factor = (
        stats.gross_profit / stats.gross_loss if stats.gross_loss > 0 else float("inf")
    )
    stats.expectancy = stats.net_pnl / stats.trades
    stats.avg_win = stats.gross_profit / stats.wins if stats.wins else 0.0
    stats.avg_loss = -stats.gross_loss / stats.losses if stats.losses else 0.0
    stats.largest_win = max(pnls)
    stats.largest_loss = min(pnls)

    r_values = [t.max_r for t in trades if t.max_r]
    if r_values:
        stats.expectancy_r = sum(r_values) / len(r_values)

    # Sharpe / Sortino from the equity curve
    if len(equity_curve) > 2:
        returns = [
            (equity_curve[i][1] - equity_curve[i - 1][1]) / equity_curve[i - 1][1]
            for i in range(1, len(equity_curve))
            if equity_curve[i - 1][1] > 0
        ]
        if returns:
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
            sd = math.sqrt(variance)
            periods = BARS_PER_YEAR.get(timeframe, 8_760)
            stats.sharpe = (mean / sd * math.sqrt(periods)) if sd > 0 else 0.0
            downside = [r for r in returns if r < 0]
            if downside:
                dmean = sum(downside) / len(downside)
                dvar = sum((r - dmean) ** 2 for r in downside) / max(len(downside) - 1, 1)
                dsd = math.sqrt(dvar)
                stats.sortino = (mean / dsd * math.sqrt(periods)) if dsd > 0 else 0.0

    if bars_in_market is not None and total_bars:
        stats.exposure_pct = bars_in_market / total_bars * 100.0
    stats.avg_bars_in_trade = 0.0  # filled by the caller when bar counts are known

    for trade in trades:
        count, pnl = stats.by_reason.get(trade.exit_reason, (0, 0.0))
        stats.by_reason[trade.exit_reason] = (count + 1, pnl + trade.net_pnl)
        count, pnl = stats.by_symbol.get(trade.symbol, (0, 0.0))
        stats.by_symbol[trade.symbol] = (count + 1, pnl + trade.net_pnl)

    return stats
