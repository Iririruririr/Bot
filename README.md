# Bot

An FX trading bot **and** a scaling bot, written in dependency-free Python.

The two bots are deliberately separate concerns:

| Bot | Job |
|---|---|
| **Trading bot** | Decides *when* to be in the market. Strategies emit signals, the risk manager sizes them, the broker fills them. |
| **Scaling bot** | Decides *how much* to have on and *when* to get out. Manages an open position by scaling in (pyramiding or averaging down) and scaling out at R multiples. |

Both run against the same engine, so a strategy can be backtested, paper-traded
and run live without changing a line of code.

```
market data ──► strategy ──► risk manager ──► scaling bot ──► broker
   (feed)      (signals)     (sizing, caps)   (adds/exits)   (paper / OANDA)
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pytest + numpy/pandas for dev only

python -m bot demo                     # offline demo, no API keys needed
```

The demo builds a synthetic EUR/USD series and runs the same strategy under four
scaling configurations, so you can see what the scaling bot actually changes.

## Commands

```
python -m bot demo                     # compare scaling modes offline
python -m bot backtest                 # backtest a strategy
python -m bot paper                    # paper trade (simulated or live OANDA)
python -m bot scale                    # run the scaling bot on its own
python -m bot grid                     # run the grid/ladder bot
python -m bot gen-data                 # write a synthetic historical CSV
python -m bot report                   # report from the local database
python -m bot strategies               # list strategies
python -m bot web --port 8000          # serve the web dashboard
```

Every command takes `--help`. Common flags:

```
--symbol EUR/USD --timeframe 1h --bars 4000 --source sim|csv|oanda
--cash 100000 --spread 1.0 --slippage 0.2 --commission 0
--risk-pct 1.0 --max-daily-loss 3 --max-drawdown 15 --max-exposure 100
```

### Backtesting

```bash
python -m bot backtest --strategy ma_crossover --bars 5000 \
    --scale-mode pyramid --tranches 3 --add-step-r 1.0 \
    --scale-out-r 1 2 3 --scale-out-frac 0.5 0.25 0.25 \
    --breakeven-at-r 1 --trail-atr 3
```

Use `--source csv --data data/EURUSD_1h.csv` to replay real historical bars
(`python -m bot gen-data` writes a synthetic CSV to get started).

### The scaling bot

```bash
# pyramid: add into strength
python -m bot backtest --scale-mode pyramid --tranches 3 --add-step-r 1.0

# DCA: average into weakness
python -m bot backtest --scale-mode dca --tranches 3 --add-step-r 0.5

# run it standalone on one manually-opened position
python -m bot scale --side buy --qty 100000 --stop-pips 40 --entry-bar 80 \
    --scale-mode dca --tranches 3 --scale-out-r 1 2 --scale-out-frac 0.5 0.5
```

Everything is expressed in **R** — the distance from entry to the first stop.
`--add-step-r 1.0` means "add one tranche for every 1R the price moves";
`--scale-out-r 1 2 3 --scale-out-frac 0.5 0.25 0.25` means "close half at 1R,
a quarter at 2R, a quarter at 3R".

After every scale-in the stop is **re-solved so total open risk stays constant**
instead of doubling with each tranche — the single most important safety
property of a scaling bot.

### The grid/ladder bot

A second flavour of "scaling bot": instead of managing one position it places a
ladder of resting limit orders around a price and lets the market scale *it* in.

```bash
python -m bot grid --centre 1.08 --levels 5 --spacing 25 --qty 10000 \
    --take-profit 20 --direction both --hard-stop 250 --max-qty 100000
```

Grid bots harvest small profits in a range and bleed in a trend, so always pair
them with `--hard-stop` or `--max-qty`.

## Web dashboard

A single-page dashboard, served by a stdlib `http.server` — the repo stays
dependency-free.

```bash
python -m bot web --port 8000
# then open http://localhost:8000
```

It binds to `0.0.0.0` by default, so it also works from a phone on the same
network, and the layout reflows down to a single column.

From the sidebar you can:

* **Run a backtest** on any strategy/scaling/risk configuration and get the
  equity curve, the full statistics block, the trade list and the scaling-bot
  activity feed.
* **Watch a live paper session** — the same engine replays a simulated feed
  bar-by-bar, so the scaling bot's adds, breakeven moves, trailing stops and
  scale-out tranches appear as they happen. Positions, tranche count, open P&L
  and the live equity curve update every bar, and you can stop the session at
  any point.

Everything is synthetic EUR/USD data, so the mechanics are real but the returns
are not a forecast.

The JSON API behind it:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | liveness + version |
| `GET` | `/api/strategies` | registered strategies and their defaults |
| `GET` | `/api/config` | current default configuration |
| `POST` | `/api/run` | start a backtest → `{job_id}` |
| `POST` | `/api/paper` | start a live replay → `{job_id}` |
| `GET` | `/api/jobs` | every job |
| `GET` | `/api/jobs/{id}` | job state, progress and latest result |
| `POST` | `/api/jobs/{id}/stop` | request an early stop |

Jobs run on their own threads, so a long replay never blocks the UI; the page
polls its job and re-renders as the result grows.

## Live trading (OANDA)

The paper and live paths share the same engine. OANDA is supported because it
offers free **practice** accounts:

```bash
export OANDA_TOKEN=your-token
export OANDA_ACCOUNT=101-001-1234567-001
export OANDA_HOST=api-fxpractice.oanda.com      # api-fxtrade.oanda.com for live

python -m bot paper --source oanda --symbol EUR_USD --strategy ma_crossover
```

Nothing is hard-coded and no credentials live in the repo. **Start on a practice
account**, and keep the risk limits on — the daily-loss and drawdown kill
switches are there for a reason.

## Configuration

Configs can be loaded from JSON (`bot.config.BotConfig`) or built entirely from
CLI flags. `bot/config.json`:

```json
{
  "broker": {"starting_cash": 100000, "spread_pips": 1.0, "slippage_pips": 0.2},
  "risk":   {"risk_per_trade_pct": 1.0, "max_daily_loss_pct": 3.0, "max_drawdown_pct": 15.0},
  "scaling": {"tranches": 3, "mode": "pyramid", "add_step_r": 1.0,
              "scale_out": [[1.0, 0.5], [2.0, 0.25], [3.0, 0.25]]}
}
```

## Layout

```
bot/
├── core/       models, risk manager, trading engine
├── brokers/    paper broker (default) + OANDA live adapter
├── data/       simulated/CSV/OANDA feeds, SQLite storage
├── strategies/ MA crossover, RSI reversion, breakout + indicators
├── scaling/    the scaling bot (scale-in/out) and the grid ladder
├── backtest/   backtest runner + performance statistics
├── web/        stdlib dashboard server + static assets
├── report.py   terminal tables and equity sparklines
├── config.py   JSON/env configuration
└── cli.py      command line interface
```

## Design notes

* **No look-ahead.** Signals are evaluated on a bar's close and filled at that
  same close, crossing the spread plus slippage.
* **Pessimistic stops.** When a bar contains both the stop and the target, the
  stop is assumed to have triggered first.
* **FIFO tranche accounting.** Partial exits report the entry price of the units
  actually sold, and the position's average cost is recomputed from what's left.
* **Currency-aware notional.** `USD/JPY` exposure is measured in USD (1 unit =
  1 USD), not in JPY — the naive version is wrong by a factor of ~150.
* **Stdlib only.** `numpy`/`pandas` are dev extras; the bot itself has no
  third-party runtime dependencies.

## Tests

```bash
python -m pytest tests/ -q
```

195 tests covering the broker (spread, stops, netting, FIFO), risk sizing for
USD-quoted and USD-based pairs, the scaling bot (add timing, scale-out ladder,
breakeven, trailing, constant-risk rebalancing), the grid ladder, the strategies,
and end-to-end backtests.

## Disclaimer

This is trading software, not trading advice. Simulated results are not a
promise of future returns. Use the paper broker and a practice account until you
trust the numbers, and never risk money you cannot afford to lose.
