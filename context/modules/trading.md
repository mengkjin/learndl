# Trading & Portfolio
**Purpose:** Portfolio construction (top-N, screening, optimized), live tracking, and backtesting. Converts alpha signals into portfolio weights and evaluates performance metrics.
**Key source paths:** `src/res/trading/`, `src/api/trading.py`, `src/api/summary.py`
**Depends on:** [[nn_models]], [[factor_engine]], [[data_pipeline]], [[project_infra]]

---

## Core Concepts

Three portfolio types, each with a dedicated manager:
- **`TrackingPort`** — live portfolio tracked in production (daily rebalance suggestions)
- **`BacktestPort`** — historical simulation of a strategy
- **`TradingPort`** — base dataclass holding the portfolio specification

Three weight construction strategies, each a `PortfolioCreator` subclass:
- **`TopStocksPortfolioCreator`** — equal-weight top-N by alpha rank
- **`ScreeningPortfolioCreator`** — rule-based screening then equal-weight
- **`OptimizedPortfolioCreator`** — constrained convex optimization

---

## API Layer

### `TradingAPI` (`src/api/trading.py`)

| Method | Purpose |
|--------|---------|
| `TradingAPI.available_ports()` | List all configured portfolio specs from `configs/trading/` |
| `TradingAPI.backtest_rebuild()` | Rebuild all backtest portfolios from scratch (full history) |
| `TradingAPI.update()` | Update live tracking portfolios with latest signals |
| `TradingAPI.Analyze(port_name)` | Run full performance analysis for a portfolio |
| `TradingAPI.Backtest(port_name)` | Run backtest for a specific portfolio config |

### `SummaryAPI` (`src/api/summary.py`)

| Function | Purpose |
|----------|---------|
| `display_account_summary()` | Print account P&L summary to console |
| `model_account_summary(model_name)` | Model-level alpha performance summary |
| `tracking_port_account_summary(port_name)` | Live portfolio P&L, turnover, Sharpe |
| `backtest_port_account_summary(port_name)` | Backtest P&L, drawdown, annual return |
| `account_summaries()` | Aggregate summary across all accounts |

---

## Core Dataclasses

### `TradingPort` (portfolio specification)
```python
@dataclass
class TradingPort:
    name:          str         # portfolio identifier
    alpha:         str         # signal source (model schedule or factor name)
    universe:      str         # stock universe: 'hs300', 'zz500', 'all_a', ...
    category:      str         # 'top' | 'screen' | 'optim'
    top_num:       int         # number of positions
    freq:          str         # rebalance: 'daily' | 'weekly' | 'monthly'
    benchmark:     str         # benchmark index (e.g., '000300.SH')
    exclusion:     list[str]   # stocks always excluded
    buffer_zone:   float|None  # buffer fraction for turnover reduction (top only)
    indus_control: bool        # apply industry neutralization
```

### `TrackingPort`
Extends `TradingPort` with live state:

| Attribute / Method | Description |
|--------------------|-------------|
| `portfolio_dir` | Directory holding current position files |
| `build(date)` | Build positions for a specific date |
| `build_portfolio(alpha_data, date)` | Compute weights from alpha, update position state |

### `BacktestPort`
Extends `TradingPort` with historical simulation:

| Attribute / Method | Description |
|--------------------|-------------|
| `build_backward(start_date, end_date)` | Simulate entire history in one pass |

---

## Portfolio Managers

### `TrackingPortfolioManager`
Manages a set of live `TrackingPort` instances.

**`update(date)` logic:**
1. Load latest alpha signals for each portfolio's `alpha` source
2. For each port call `build_portfolio(alpha_data, date)`
3. Generate trade suggestions (current holdings → target weights)
4. Write position files to `portfolio_dir`

**Buffer zone logic** (reduces turnover):
```
if buffer_zone is set (e.g., 0.2):
    cutoff_rank = top_num
    buffer_low  = top_num * (1 - buffer_zone)   # must enter if rank ≤ this
    buffer_high = top_num * (1 + buffer_zone)   # must exit if rank > this

    new entrants: rank ≤ buffer_low  → enter regardless
    held stocks:  rank ≤ buffer_high → stay if currently held
    exits:        rank > buffer_high → forced exit

Without buffer_zone: straightforward top-N by alpha rank.
```

### `BacktestPortfolioManager`

| Method | Purpose |
|--------|---------|
| `available_ports()` | List configured backtest port names |
| `analyze(port_name)` | Run analysis tasks on completed backtest |
| `rebuild(port_name)` | Full historical rebuild from scratch |
| `update(port_name, date)` | Incremental update — add latest date only |

---

## Portfolio Construction — `PortfolioBuilder`

Central class that orchestrates weight computation.

| Method | Purpose |
|--------|---------|
| `setup(trading_port, date)` | Load alpha, universe, constraints for a date |
| `build()` | Dispatch to the appropriate `PortfolioCreator` |
| `accounting(weights, date)` | Record NAV, turnover, positions to storage |

### `TopStocksPortfolioCreator`
Equal-weight top-N by descending alpha rank, with optional buffer zone (see above).

### `ScreeningPortfolioCreator`
Apply boolean screens (minimum liquidity, exclude ST stocks, etc.) then equal-weight survivors.

### `OptimizedPortfolioCreator`
Constrained convex optimization. Supports three problem types:

| Type | Formulation |
|------|-------------|
| `linprog` | Linear programming — maximize alpha subject to linear constraints |
| `quadprog` | Quadratic programming — maximize alpha − λ·risk |
| `socp` | Second-order cone programming — tracking error constraint |

Three solver backends:
- `mosek` — commercial, fastest for large problems
- `cvxopt` — open-source QP solver
- `cvxpy` — modeling layer over multiple solvers (recommended default)

Typical constraints:
- Long-only or long-short bounds
- Industry exposure neutralization (`indus_control: true`)
- Turnover limit (max weight change per rebalance)
- Factor exposure limits (from risk model covariance)
- Stock weight bounds (min/max per position)

---

## Performance Analysis Tasks

Run via `TradingAPI.Analyze(port_name)`. Each task produces a plot and metrics dict:

| Task | Output |
|------|--------|
| `FrontFace` | Summary page: NAV curve, key stats |
| `Perf_Curve` | Cumulative return vs. benchmark |
| `Perf_Excess` | Excess return (alpha) cumulative curve |
| `Drawdown` | Maximum drawdown chart |
| `Perf_Year` | Annual return bar chart |

---

## Configuration — `configs/trading/trading_port.yaml`

```yaml
portfolios:
  - name: top50_lgbm
    alpha: lgbm_of_factors        # signal source (model schedule name)
    universe: hs300
    category: top
    top_num: 50
    freq: daily
    benchmark: 000300.SH
    exclusion: []
    buffer_zone: 0.2              # 20% buffer around rank cutoff
    indus_control: false

  - name: optim100_nn
    alpha: gru_ret1d
    universe: zz500
    category: optim
    top_num: 100
    freq: weekly
    benchmark: 000905.SH
    exclusion: []
    buffer_zone: null
    indus_control: true
```

---

## Data Flow

```
configs/trading/trading_port.yaml
        ↓
TradingAPI / BacktestPortfolioManager
        ↓
PortfolioBuilder.setup()
  → alpha signals: PATH.prediction/{alpha}/{date}.feather
  → universe: DataAPI / TRADE.get_universe()
  → risk model: FactorAPI (for optim portfolios)
        ↓
PortfolioCreator (Top / Screen / Optim)
  → weights: np.ndarray (N_secid,)
        ↓
PortfolioBuilder.accounting()
  → positions: PATH.portfolio/{port_name}/{date}.feather
        ↓
SummaryAPI / analysis tasks
  → performance metrics + plots
```

---

## Common Patterns / Gotchas

- `TradingAPI.backtest_rebuild()` re-runs full history — only use when signal data has materially changed; it is slow
- Incremental updates go through `update()`, not `rebuild()`
- `buffer_zone` is a fraction of `top_num` — `0.2` means ±20% around the cutoff rank; set `null` to disable
- `SummaryAPI` is the correct way to compute performance metrics — do not reimplement Sharpe/drawdown calculations
- For `optim` portfolios, ensure the risk model is up to date — run the risk factor update scripts first
- `indus_control: true` requires industry classification data from `DataAPI` / `INFO.get_industry()`
- Portfolio weight files use feather format via `df_io` from [[project_infra]]
- Portfolio config lives at `configs/setting/trading_port.yaml` (not `configs/trading/`)
- See [[workflows/strategy_dev]] for the end-to-end flow of testing a new alpha signal
