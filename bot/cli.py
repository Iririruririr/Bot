"""Command line interface.

    python -m bot demo                  # offline demo: compare scaling modes
    python -m bot backtest              # backtest a strategy on sim/CSV data
    python -m bot paper                 # paper-trade (simulated or live OANDA)
    python -m bot scale                 # run the scaling bot on its own
    python -m bot grid                  # run the grid/ladder bot
    python -m bot gen-data              # write a synthetic historical CSV
    python -m bot report                # report from the local database
    python -m bot strategies            # list available strategies
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional, Sequence

from bot.backtest.runner import run_backtest
from bot.brokers.paper import BrokerConfig, PaperBroker
from bot.config import BotConfig, has_oanda_credentials, oanda_credentials
from bot.core.engine import Engine, EngineConfig
from bot.core.models import Order, OrderType, Side, spec_for
from bot.core.risk import RiskConfig, RiskManager
from bot.data.feed import (
    TIMEFRAME_SECONDS,
    CsvFeed,
    DataFeed,
    SimConfig,
    SimulatedFeed,
    generate_csv,
)
from bot.data.storage import SQLiteStore
from bot.report import (
    comparison_report,
    equity_report,
    scaling_report,
    summary_report,
    table,
    trades_report,
)
from bot.scaling.engine import ScaleOutLeg, ScalingConfig, ScalingEngine
from bot.scaling.ladder import GridLadder, LadderConfig
from bot.strategies.base import available, build

log = logging.getLogger("bot")


# --------------------------------------------------------------------------- #
# feed / config helpers
# --------------------------------------------------------------------------- #


def build_feed(args) -> DataFeed:
    source = getattr(args, "source", "sim")
    if source == "csv":
        return CsvFeed(args.data, symbol=args.symbol, timeframe=args.timeframe)
    if source == "oanda":
        from bot.data.oanda import OandaClient, OandaFeed

        client = OandaClient()
        instrument = args.symbol.replace("/", "_")
        return OandaFeed(client, instrument, timeframe=args.timeframe, count=args.bars)
    return SimulatedFeed(
        SimConfig(
            symbol=args.symbol,
            bars=args.bars,
            timeframe=args.timeframe,
            seed=args.seed,
            start_price=getattr(args, "start_price", None) or 1.0800,
            vol_pips=args.vol_pips,
        )
    )


def scaling_from_args(args) -> ScalingConfig:
    mode = getattr(args, "scale_mode", "off") or "off"
    if mode == "off":
        return ScalingConfig(tranches=1)
    if len(args.scale_out_r) != len(args.scale_out_frac):
        raise SystemExit("--scale-out-r and --scale-out-frac must have the same length")
    legs = tuple(
        ScaleOutLeg(r, fraction)
        for r, fraction in zip(args.scale_out_r, args.scale_out_frac)
    )
    default_legs = (ScaleOutLeg(1.0, 0.5), ScaleOutLeg(2.0, 0.5))
    if mode == "pyramid":
        cfg = ScalingConfig.pyramid(
            tranches=args.tranches,
            step_r=args.add_step_r,
            scale_out=legs or default_legs,
            breakeven_at_r=args.breakeven_at_r,
        )
    elif mode == "dca":
        cfg = ScalingConfig.dca(
            tranches=args.tranches,
            step_r=args.add_step_r,
            scale_out=legs or default_legs,
            breakeven_at_r=args.breakeven_at_r,
        )
    else:
        raise SystemExit(f"unknown scale mode {mode!r}")
    cfg.add_size_mult = args.add_size_mult
    if args.trail_atr:
        cfg.trail_atr_mult = args.trail_atr
        cfg.trail_start_r = args.trail_start_r
    if args.time_stop:
        cfg.time_stop_bars = args.time_stop
    return cfg


def broker_from_args(args, starting_cash: Optional[float] = None) -> BrokerConfig:
    return BrokerConfig(
        starting_cash=starting_cash if starting_cash is not None else args.cash,
        spread_pips=args.spread,
        slippage_pips=args.slippage,
        commission_per_million=args.commission,
        lot_size=args.lot_size,
    )


def risk_from_args(args) -> RiskConfig:
    return RiskConfig(
        risk_per_trade_pct=args.risk_pct,
        max_daily_loss_pct=args.max_daily_loss,
        max_drawdown_pct=args.max_drawdown,
        max_open_exposure_pct=args.max_exposure,
        max_positions=args.max_positions,
        lot_size=args.lot_size,
    )


def storage_from_args(args) -> Optional[SQLiteStore]:
    if getattr(args, "no_db", False):
        return None
    try:
        return SQLiteStore(args.db)
    except Exception as exc:  # pragma: no cover - disk issues
        log.warning("storage disabled: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_demo(args) -> int:
    """Offline demo: the same trading bot under four scaling configurations."""
    print("\nBuilding a synthetic EUR/USD 1h series "
          f"({args.bars:,} bars, seed {args.seed}) ...")
    feed = SimulatedFeed(
        SimConfig(symbol=args.symbol, bars=args.bars, timeframe="1h", seed=args.seed, start_price=1.0800)
    )
    bars = list(feed)
    print(f"  {bars[0].time:%Y-%m-%d} -> {bars[-1].time:%Y-%m-%d}, "
          f"{bars[0].close:.5f} -> {bars[-1].close:.5f}")

    runs = [
        ("trading bot only", ScalingConfig(tranches=1)),
        ("pyramid scale-in", ScalingConfig.pyramid(tranches=3, step_r=1.0, breakeven_at_r=1.0)),
        ("dca scale-in", ScalingConfig.dca(tranches=3, step_r=0.5, breakeven_at_r=1.0)),
        (
            "pyramid + trail",
            ScalingConfig.pyramid(tranches=3, step_r=1.0, breakeven_at_r=1.0, trail_atr_mult=3.0),
        ),
    ]

    results = []
    details = {}
    for name, scaling in runs:
        result = run_backtest(
            feed=feed,
            strategy=build(args.strategy),
            scaling=scaling,
            risk=risk_from_args(args),
            broker=broker_from_args(args),
            storage=None,
        )
        results.append((name, result.stats))
        details[name] = result
        print(f"  ran {name:<18} -> {result.stats.trades:>4} trades, "
              f"net {result.stats.net_pnl:>12,.2f}")

    print(comparison_report(results))

    best_name, best_stats = max(results, key=lambda item: item[1].sharpe)
    best = details[best_name]
    print(f"\nBest configuration by Sharpe: {best_name}")
    print(summary_report(best.stats, title=f"Detail - {best_name}", currency="$"))
    print(equity_report(best.equity_curve))
    print(scaling_report(best.scaling_stats, best.events))
    print(trades_report(best.trades, limit=15))
    return 0


def cmd_backtest(args) -> int:
    feed = build_feed(args)
    strategy = build(args.strategy, **parse_params(args.params))
    scaling = scaling_from_args(args)
    storage = storage_from_args(args)

    print(f"\nBacktesting {strategy} on {feed.symbol} {feed.timeframe} "
          f"({len(feed):,} bars, source={args.source})")
    print(f"Scaling bot: {scaling.mode} x{scaling.tranches} tranches, "
          f"step {scaling.add_step_r}R, scale-out "
          f"{[(leg.r, leg.fraction) for leg in scaling.scale_out]}")

    result = run_backtest(
        feed=feed,
        strategy=strategy,
        scaling=scaling,
        risk=risk_from_args(args),
        broker=broker_from_args(args),
        storage=storage,
    )
    print(summary_report(result.stats, title=f"{strategy.name} on {feed.symbol}"))
    print(equity_report(result.equity_curve))
    print(scaling_report(result.scaling_stats, result.events))
    print(trades_report(result.trades, limit=args.trades))
    return 0


def cmd_paper(args) -> int:
    """Paper-trade against a simulated feed or live OANDA prices."""
    strategy = build(args.strategy, **parse_params(args.params))
    scaling = scaling_from_args(args)
    broker = broker_from_args(args)
    risk = RiskManager(risk_from_args(args), broker.starting_cash)
    storage = storage_from_args(args)
    paper_broker = PaperBroker(broker)
    scaling_engine = ScalingEngine(scaling, paper_broker, broker.account_ccy, risk=risk)
    engine = Engine(
        broker=paper_broker,
        strategies=[strategy],
        risk=risk,
        scaling=scaling_engine,
        storage=storage,
        config=EngineConfig(exit_on_flip=not args.no_flip),
    )
    scaling_engine.on_event = engine.emit

    if args.source == "oanda":
        if not has_oanda_credentials():
            print("OANDA credentials missing. Set OANDA_TOKEN and OANDA_ACCOUNT "
                  "(practice accounts are free) or use --source sim.")
            return 2
        from bot.data.oanda import OandaClient, OandaFeed

        client = OandaClient(**oanda_credentials())
        instrument = args.symbol.replace("/", "_")
        feed = OandaFeed(client, instrument, timeframe=args.timeframe, count=args.bars)
        print(f"Live paper trading on OANDA ({client.host}, account {client.account})")
        seen = 0
        polled = 0
        while polled < args.max_polls:
            candles = feed.refresh()
            for candle in candles[seen:]:
                engine.on_candle(candle)
            seen = len(candles)
            polled += 1
            _print_status(engine, paper_broker, args.symbol, live=True)
            time.sleep(args.interval)
        return 0

    feed = build_feed(args)
    print(f"Paper trading on simulated {feed.symbol} {feed.timeframe} "
          f"({args.bars:,} bars)")
    stride = max(args.bars // 40, 1)
    for index, candle in enumerate(feed):
        engine.on_candle(candle)
        if index % stride == 0:
            _print_status(engine, paper_broker, args.symbol)
    _print_status(engine, paper_broker, args.symbol)
    print(summary_report_from_engine(engine, broker.config.starting_cash, feed.timeframe, scaling_engine))
    print(trades_report(engine.trades(), limit=args.trades))
    return 0


def _print_status(engine: Engine, broker: PaperBroker, symbol: str, live: bool = False) -> None:
    position = broker.positions().get(symbol)
    equity = broker.equity()
    tag = "live " if live else ""
    if position is not None and position.is_open:
        print(
            f"  {tag}equity {equity:>12,.2f} | {symbol} {position.side.value} "
            f"{position.qty:>10,.0f} @ {position.avg_price:.5f} | stop "
            f"{position.stop_loss if position.stop_loss else float('nan'):.5f} | "
            f"unrealised {broker.unrealized_pnl():>10,.2f} | trades {len(broker.closed_trades())}"
        )
    else:
        print(f"  {tag}equity {equity:>12,.2f} | flat | trades {len(broker.closed_trades())}")


def cmd_scale(args) -> int:
    """Run the scaling bot on its own, managing one manually opened position."""
    feed = build_feed(args)
    broker = PaperBroker(broker_from_args(args))
    scaling = scaling_from_args(args)
    if scaling.tranches <= 1 and not scaling.scale_out:
        print("Nothing for the scaling bot to do - pass --scale-mode pyramid|dca "
              "and/or --scale-out-r/--scale-out-frac.")
        return 2

    risk = RiskManager(risk_from_args(args), broker.config.starting_cash)
    scaling_engine = ScalingEngine(
        scaling, broker, broker.config.account_ccy, risk=risk
    )
    engine = Engine(broker=broker, strategies=[], scaling=scaling_engine, risk=risk)
    scaling_engine.on_event = engine.emit

    side = Side.BUY if args.side == "buy" else Side.SELL
    spec = spec_for(args.symbol)
    entered = False
    print(f"\nScaling bot on {args.symbol}: entry at bar {args.entry_bar}, "
          f"{args.qty:,.0f} units {side.value}, stop {args.stop_pips:.0f} pips, "
          f"mode {scaling.mode} x{scaling.tranches}")

    for index, candle in enumerate(feed):
        engine.on_candle(candle)
        if not entered and index >= args.entry_bar:
            stop = candle.close - side.sign * args.stop_pips * spec.pip_size
            fill = broker.submit(
                Order(
                    symbol=args.symbol,
                    side=side,
                    qty=args.qty,
                    order_type=OrderType.MARKET,
                    stop_loss=stop,
                    tag="manual-entry",
                )
            )
            entered = fill is not None
            if entered:
                print(f"  entered {side.value} {fill.qty:,.0f} @ {fill.price:.5f} "
                      f"(stop {stop:.5f}, R = {args.stop_pips:.0f} pips)")
        if index % max(args.bars // 10, 1) == 0:
            _print_status(engine, broker, args.symbol)

    print(summary_report_from_engine(engine, broker.config.starting_cash, feed.timeframe, scaling_engine))
    print(scaling_report(scaling_engine.stats, engine.events))
    print(trades_report(engine.trades(), limit=args.trades))
    return 0


def cmd_grid(args) -> int:
    """Run the grid/ladder bot."""
    feed = build_feed(args)
    broker = PaperBroker(broker_from_args(args))
    engine = Engine(broker=broker, strategies=[])
    ladder = GridLadder(
        LadderConfig(
            centre_price=args.centre,
            levels=args.levels,
            spacing_pips=args.spacing,
            qty_per_level=args.qty,
            take_profit_pips=args.take_profit,
            direction=args.direction,
            stop_loss_pips=args.hard_stop,
            max_total_qty=args.max_qty,
        ),
        broker,
        symbol=args.symbol,
        on_event=engine.emit,
    )
    ladder.build()
    print(f"\nGrid ladder on {args.symbol}: {args.levels} rungs/side, "
          f"{args.spacing:.0f} pip spacing, {args.qty:,.0f} units per rung, "
          f"TP {args.take_profit:.0f} pips")

    for candle in feed:
        engine.on_candle(candle)
        ladder.on_candle(candle)
        if candle.time.hour == 0:  # daily progress line
            position = broker.positions().get(args.symbol)
            qty = position.qty if position else 0.0
            print(f"  {candle.time:%Y-%m-%d} {candle.close:.5f} | open qty {qty:>10,.0f} "
                  f"| equity {broker.equity():>12,.2f} | trades {len(broker.closed_trades())}")

    print("\n=== Grid ladder summary ===")
    print(table(["Metric", "Value"], [(k, f"{v:,}" if isinstance(v, int) else f"{v:,.2f}")
                                      for k, v in ladder.summary().items()]))
    print(trades_report(broker.closed_trades(), limit=args.trades))
    return 0


def cmd_gen_data(args) -> int:
    path = generate_csv(
        args.out,
        symbol=args.symbol,
        bars=args.bars,
        timeframe=args.timeframe,
        seed=args.seed,
    )
    print(f"Wrote {path}")
    return 0


def cmd_report(args) -> int:
    store = SQLiteStore(args.db)
    trades = store.trades()
    curve = store.equity_curve()
    if not trades or not curve:
        print("No data yet - run a backtest or paper session first.")
        return 0
    from bot.backtest.stats import compute_stats

    stats = compute_stats(trades, curve, curve[0][1])
    print(summary_report(stats, title="Stored session report"))
    print(equity_report(curve))
    print(trades_report(trades, limit=args.trades))
    return 0


def cmd_strategies(args) -> int:
    rows = []
    for name in available():
        strategy = build(name)
        rows.append((name, str(strategy.warmup), str(sorted(strategy.params))))
    print(table(["Strategy", "Warmup", "Default params"], rows))
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def parse_params(raw: Optional[str]) -> dict:
    """Parse ``key=value,key=value`` into a dict with numeric coercion."""
    if not raw:
        return {}
    out = {}
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key, value = key.strip(), value.strip()
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def summary_report_from_engine(engine: Engine, starting_equity: float, timeframe: str, scaling_engine) -> str:
    from bot.backtest.stats import compute_stats

    stats = compute_stats(
        engine.trades(),
        engine.equity_curve,
        starting_equity,
        timeframe=timeframe,
        bars_in_market=engine.bars_in_market,
        total_bars=sum(engine.bar_index.values()),
    )
    return summary_report(stats, title="Session summary")


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bot",
        description="FX trading bot + scaling bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", default="data/bot.db", help="SQLite database path")
    parser.add_argument("--config", default=None, help="JSON config file (see config/default.json)")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, bars_default: int = 4000):
        # SUPPRESS keeps the top-level --db / --config values unless overridden here
        p.add_argument("--db", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        p.add_argument("--symbol", default="EUR/USD")
        p.add_argument("--timeframe", default="1h", choices=list(TIMEFRAME_SECONDS))
        p.add_argument("--bars", type=int, default=bars_default)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--vol-pips", dest="vol_pips", type=float, default=12.0)
        p.add_argument("--start-price", dest="start_price", type=float, default=1.0800)
        p.add_argument("--source", default="sim", choices=["sim", "csv", "oanda"])
        p.add_argument("--data", default="data/EURUSD_1h.csv", help="CSV path when --source csv")
        p.add_argument("--cash", type=float, default=100_000.0)
        p.add_argument("--spread", type=float, default=1.0, help="broker spread in pips")
        p.add_argument("--slippage", type=float, default=0.2, help="slippage in pips")
        p.add_argument("--commission", type=float, default=0.0, help="commission per 1M notional")
        p.add_argument("--lot-size", dest="lot_size", type=float, default=1_000.0)
        p.add_argument("--trades", type=int, default=25, help="trades to print")
        p.add_argument("--no-db", action="store_true", help="do not write to SQLite")

    def add_risk(p):
        p.add_argument("--risk-pct", dest="risk_pct", type=float, default=1.0)
        p.add_argument("--max-daily-loss", dest="max_daily_loss", type=float, default=3.0)
        p.add_argument("--max-drawdown", dest="max_drawdown", type=float, default=15.0)
        p.add_argument("--max-exposure", dest="max_exposure", type=float, default=100.0)
        p.add_argument("--max-positions", dest="max_positions", type=int, default=4)

    def add_scaling(p):
        p.add_argument("--scale-mode", dest="scale_mode", default="off",
                       choices=["off", "pyramid", "dca"])
        p.add_argument("--tranches", type=int, default=3)
        p.add_argument("--add-step-r", dest="add_step_r", type=float, default=1.0)
        p.add_argument("--add-size-mult", dest="add_size_mult", type=float, default=1.0)
        p.add_argument("--scale-out-r", dest="scale_out_r", type=float, nargs="*", default=[])
        p.add_argument("--scale-out-frac", dest="scale_out_frac", type=float, nargs="*", default=[])
        p.add_argument("--breakeven-at-r", dest="breakeven_at_r", type=float, default=None)
        p.add_argument("--trail-atr", dest="trail_atr", type=float, default=None)
        p.add_argument("--trail-start-r", dest="trail_start_r", type=float, default=1.0)
        p.add_argument("--time-stop", dest="time_stop", type=int, default=None)

    demo = sub.add_parser("demo", help="offline demo comparing scaling modes")
    demo.add_argument("--strategy", default="ma_crossover")
    add_common(demo)
    add_risk(demo)
    demo.set_defaults(func=cmd_demo)

    backtest = sub.add_parser("backtest", help="backtest a strategy")
    backtest.add_argument("--strategy", default="ma_crossover")
    backtest.add_argument("--params", default=None, help="strategy params, e.g. fast=10,slow=30")
    add_common(backtest)
    add_risk(backtest)
    add_scaling(backtest)
    backtest.set_defaults(func=cmd_backtest)

    paper = sub.add_parser("paper", help="paper trade")
    paper.add_argument("--strategy", default="ma_crossover")
    paper.add_argument("--params", default=None)
    paper.add_argument("--interval", type=float, default=30.0, help="seconds between OANDA polls")
    paper.add_argument("--max-polls", dest="max_polls", type=int, default=1000)
    paper.add_argument("--no-flip", dest="no_flip", action="store_true")
    add_common(paper)
    add_risk(paper)
    add_scaling(paper)
    paper.set_defaults(func=cmd_paper)

    scale = sub.add_parser("scale", help="run the scaling bot on its own")
    scale.add_argument("--side", default="buy", choices=["buy", "sell"])
    scale.add_argument("--qty", type=float, default=100_000.0)
    scale.add_argument("--stop-pips", dest="stop_pips", type=float, default=40.0)
    scale.add_argument("--entry-bar", dest="entry_bar", type=int, default=60)
    add_common(scale)
    add_risk(scale)
    add_scaling(scale)
    scale.set_defaults(func=cmd_scale)

    grid = sub.add_parser("grid", help="run the grid/ladder bot")
    grid.add_argument("--centre", type=float, default=1.0800)
    grid.add_argument("--levels", type=int, default=5)
    grid.add_argument("--spacing", type=float, default=20.0)
    grid.add_argument("--qty", type=float, default=10_000.0)
    grid.add_argument("--take-profit", dest="take_profit", type=float, default=15.0)
    grid.add_argument("--direction", default="both", choices=["buy", "sell", "both"])
    grid.add_argument("--hard-stop", dest="hard_stop", type=float, default=0.0)
    grid.add_argument("--max-qty", dest="max_qty", type=float, default=0.0)
    add_common(grid)
    grid.set_defaults(func=cmd_grid)

    gen = sub.add_parser("gen-data", help="generate synthetic historical CSV")
    gen.add_argument("--out", default="data/EURUSD_1h.csv")
    add_common(gen, bars_default=5000)
    gen.set_defaults(func=cmd_gen_data)

    report = sub.add_parser("report", help="report from the database")
    report.add_argument("--trades", type=int, default=25)
    report.set_defaults(func=cmd_report)

    strategies = sub.add_parser("strategies", help="list strategies")
    strategies.set_defaults(func=cmd_strategies)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if getattr(args, "config", None):
        _apply_config(args, argv)
    return args.func(args)


# CLI flag -> (attribute, config value getter).  Used by --config so a JSON file
# supplies the defaults for anything not passed explicitly on the command line.
_CONFIG_FLAGS = {
    "--cash": ("cash", lambda c: c.broker.starting_cash),
    "--spread": ("spread", lambda c: c.broker.spread_pips),
    "--slippage": ("slippage", lambda c: c.broker.slippage_pips),
    "--commission": ("commission", lambda c: c.broker.commission_per_million),
    "--lot-size": ("lot_size", lambda c: c.broker.lot_size),
    "--risk-pct": ("risk_pct", lambda c: c.risk.risk_per_trade_pct),
    "--max-daily-loss": ("max_daily_loss", lambda c: c.risk.max_daily_loss_pct),
    "--max-drawdown": ("max_drawdown", lambda c: c.risk.max_drawdown_pct),
    "--max-exposure": ("max_exposure", lambda c: c.risk.max_open_exposure_pct),
    "--max-positions": ("max_positions", lambda c: c.risk.max_positions),
    "--tranches": ("tranches", lambda c: c.scaling.tranches),
    "--add-step-r": ("add_step_r", lambda c: c.scaling.add_step_r),
    "--add-size-mult": ("add_size_mult", lambda c: c.scaling.add_size_mult),
    "--breakeven-at-r": ("breakeven_at_r", lambda c: c.scaling.breakeven_at_r),
    "--trail-atr": ("trail_atr", lambda c: c.scaling.trail_atr_mult),
    "--time-stop": ("time_stop", lambda c: c.scaling.time_stop_bars),
    "--cooldown-bars": ("cooldown_bars", lambda c: c.engine.cooldown_bars),
}


def _apply_config(args, argv: Sequence[str]) -> None:
    """Fill unset CLI flags from a JSON config file."""
    config = BotConfig.load(args.config)
    passed = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    for flag, (attr, getter) in _CONFIG_FLAGS.items():
        if flag not in passed and hasattr(args, attr):
            setattr(args, attr, getter(config))
    if "--scale-out-r" not in passed and config.scaling.scale_out:
        args.scale_out_r = [leg.r for leg in config.scaling.scale_out]
        args.scale_out_frac = [leg.fraction for leg in config.scaling.scale_out]
    # a config with more than one tranche implies the scaling bot is on
    if "--scale-mode" not in passed and config.scaling.tranches > 1 and getattr(args, "scale_mode", "off") == "off":
        args.scale_mode = config.scaling.mode


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
