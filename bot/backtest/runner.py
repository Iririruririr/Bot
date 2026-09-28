"""Backtest runner - drives the engine over a historical (or simulated) feed."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

from bot.brokers.paper import BrokerConfig, PaperBroker
from bot.core.engine import Engine, EngineConfig
from bot.core.models import Trade
from bot.core.risk import RiskConfig, RiskManager
from bot.scaling.engine import ScalingConfig, ScalingEngine
from bot.strategies.base import Strategy
from bot.data.feed import DataFeed
from bot.data.storage import SQLiteStore
from bot.backtest.stats import Stats, compute_stats


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    stats: Optional[Stats] = None
    events: List[dict] = field(default_factory=list)
    scaling_stats: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else 0.0


def run_backtest(
    feed: DataFeed,
    strategy: Strategy,
    scaling: Optional[ScalingConfig] = None,
    risk: Optional[RiskConfig] = None,
    broker: Optional[BrokerConfig] = None,
    engine_config: Optional[EngineConfig] = None,
    storage: Optional[SQLiteStore] = None,
    on_event=None,
) -> BacktestResult:
    """Run one strategy over one feed and return trades + equity curve + stats."""
    broker_config = broker or BrokerConfig()
    paper_broker = PaperBroker(broker_config)
    risk_manager = RiskManager(risk or RiskConfig(), broker_config.starting_cash)
    scaling_engine = ScalingEngine(
        config=scaling or ScalingConfig(),
        broker=paper_broker,
        account_ccy=broker_config.account_ccy,
        risk=risk_manager,
    )
    engine = Engine(
        broker=paper_broker,
        strategies=[strategy],
        risk=risk_manager,
        scaling=scaling_engine,
        storage=storage,
        config=engine_config or EngineConfig(),
        on_event=on_event,
    )
    # the scaling bot reports through the same event log as everything else
    scaling_engine.on_event = engine.emit

    bars = list(feed)
    bars_in_market = 0
    for candle in bars:
        engine.on_candle(candle)
        if paper_broker.positions().get(candle.symbol) is not None:
            bars_in_market += 1

    result = BacktestResult(
        trades=paper_broker.closed_trades(),
        equity_curve=engine.equity_curve,
        events=engine.events,
        scaling_stats=dict(scaling_engine.stats),
        meta={
            "strategy": strategy.name,
            "strategy_params": dict(strategy.params),
            "symbol": feed.symbol,
            "timeframe": feed.timeframe,
            "bars": len(bars),
            "scaling": (scaling or ScalingConfig()).to_dict(),
            "starting_equity": broker_config.starting_cash,
        },
    )
    result.stats = compute_stats(
        result.trades,
        result.equity_curve,
        broker_config.starting_cash,
        timeframe=feed.timeframe,
        bars_in_market=bars_in_market,
        total_bars=len(bars),
    )
    return result
