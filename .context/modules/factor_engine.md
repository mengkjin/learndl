# Factor Engine
**Purpose:** Define, calculate, normalize, and test alpha factors. Auto-registration of factor subclasses via metaclass. Multi-category factor hierarchy.
**Key source paths:** `src/res/factor/`
**Depends on:** [[data_pipeline]], [[project_infra]]

---

## Key Concepts

- **`FactorCalculator`** — abstract base; every concrete factor subclasses it and is auto-registered by name
- **`FactorAPI`** — stable public interface for calculate / update / test / normalize
- **18 category base classes** — each factor inherits from one category (value, momentum, quality, etc.)
- **`StockFactor`** — normalized, portfolio-ready factor value
- **`FactorTestAPI`** — 13 calculator types for backtesting factors

---

## `FactorAPI` — public interface

```python
from src.api.factor import FactorAPI

FactorAPI.calculate(factor_name, dates, universe)   # compute factor values → DataBlock
FactorAPI.update(factor_name, start, end)           # compute and save to PATH.factor
FactorAPI.test(factor_name, test_type, **kwargs)    # run factor test
FactorAPI.normalize(factor_name, dates, universe)   # return StockFactor (normalized)
FactorAPI.available()                               # list all registered factor names
```

`FactorAPI` has sub-namespaces mirroring the test framework:

```python
FactorAPI.Calc          # FactorCalculator access
FactorAPI.Test          # FactorTestAPI access
FactorAPI.Norm          # normalization pipeline access
FactorAPI.Port          # factor-based portfolio construction
```

---

## `FactorCalculator` — abstract base (metaclass auto-registration)

Every subclass is automatically registered by its class name when the module is imported.

```python
from src.factor.calculator import FactorCalculator

class MyFactor(MomentumBase):          # inherit from a category base
    name = 'my_momentum_factor'
    params = {'window': 20}

    def calculate(self, data: DataBlock) -> DataBlock:
        close = TRADE.get_close(data.date, data.secid)
        return TsRoller(self.params['window']).mean(close)
```

Key class-level attributes all `FactorCalculator` subclasses have:
| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Registered name (used in configs and FactorAPI calls) |
| `category` | str | Auto-set from the category base class |
| `params` | dict | Default hyperparameters (overridden by `configs/factor/`) |
| `dependencies` | list[str] | Other factor names this factor depends on |
| `freq` | str | Calculation frequency: `'daily'`, `'monthly'` |

Key class methods (inherited, not overridden):
| Method | Description |
|--------|-------------|
| `FactorCalculator.get(name)` | Return the registered calculator class for `name` |
| `FactorCalculator.list()` | Return all registered factor names |
| `FactorCalculator.from_config(cfg)` | Instantiate from a config dict |

The single method subclasses must implement:
```python
def calculate(self, data: DataBlock) -> DataBlock:
    ...
```

---

## 18 Category Base Classes

Each lives in `src/factor/categories/`. Inheriting from one sets `self.category` automatically.

| Base Class | Category Name | Typical Signal Source |
|------------|--------------|----------------------|
| `ValueBase` | `value` | P/E, P/B, EV/EBITDA from financial statements |
| `MomentumBase` | `momentum` | Price momentum, return reversal |
| `QualityBase` | `quality` | ROE, earnings quality, accruals |
| `GrowthBase` | `growth` | Revenue/earnings growth rates |
| `LiquidityBase` | `liquidity` | Turnover, Amihud illiquidity |
| `SizeBase` | `size` | Market cap, free-float cap |
| `VolatilityBase` | `volatility` | Realized vol, skewness, kurtosis |
| `TechnicalBase` | `technical` | TA indicators (RSI, MACD, Bollinger) |
| `MicrostructureBase` | `microstructure` | Order flow, bid-ask, auction data |
| `SentimentBase` | `sentiment` | Analyst revisions, short interest |
| `EstimateBase` | `estimate` | Forward EPS, target price |
| `FinancialBase` | `financial` | Balance sheet ratios |
| `FundamentalBase` | `fundamental` | Mixed fundamental signals |
| `AlternativeBase` | `alternative` | Non-price alternative data |
| `CompositeBase` | `composite` | Combinations of other factors |
| `RiskBase` | `risk` | Risk model inputs (Barra-style) |
| `StyleBase` | `style` | Style tilts (growth/value/quality blend) |
| `CustomBase` | `custom` | Catch-all for non-standard factors |

---

## `StockFactorHierarchy`

Registry of all registered factors, organized by category. Useful for bulk operations:

```python
from src.factor.registry import StockFactorHierarchy

StockFactorHierarchy.all()                      # all registered FactorCalculator classes
StockFactorHierarchy.by_category('momentum')    # all momentum factors
StockFactorHierarchy.names()                    # list of all factor name strings
```

---

## `StockFactor` — normalized factor values

Output of `FactorAPI.normalize()`. Wraps a `DataBlock` with normalization metadata.

```python
sf = FactorAPI.normalize('my_factor', dates, universe)

sf.data           # DataBlock: (N_secid × N_date × 1 × 1), z-scored cross-sectionally
sf.ic             # pd.Series: IC time series
sf.icir           # float: IC information ratio
sf.name           # factor name
sf.pipeline       # list of PreProcessor names applied
```

Default normalization pipeline (configurable in `configs/factor/`):
1. `fillna_ffill`
2. `winsorize_cs` (1%/99%)
3. `neutralize_industry`
4. `neutralize_size`
5. `zscore_cs`

---

## `FactorTestAPI` — 13 calculator types

Used via `FactorAPI.test(factor_name, test_type, ...)`. Runs performance tests and writes results to `PATH.factor / 'tests/'`.

| Test Type | Key | Description |
|-----------|-----|-------------|
| IC analysis | `'ic'` | Pearson/Spearman IC, IC decay |
| Rank IC | `'rank_ic'` | Rank IC time series and ICIR |
| Quantile return | `'quantile'` | Equal-weight returns by factor quantile |
| Long-short | `'ls'` | Top-minus-bottom portfolio return |
| Turnover | `'turnover'` | Portfolio turnover at each rebalance |
| Decay | `'decay'` | IC decay over lag periods |
| Sector neutralized | `'sector_ic'` | IC computed within sector groups |
| Factor correlation | `'factor_corr'` | Cross-factor correlation matrix |
| Industry exposure | `'industry'` | Factor exposure by industry |
| Size exposure | `'size'` | Factor loading on size |
| Attribution | `'attribution'` | Return attribution to style/industry |
| Coverage | `'coverage'` | % of universe with non-NaN values |
| Stability | `'stability'` | Auto-correlation of factor values |

Run all standard tests:
```python
FactorAPI.Test.run_all(factor_name, dates, universe)
```

---

## Portfolio Optimization Config (factor-based)

When `FactorAPI.Port` builds a factor-based portfolio, the optimization config structure mirrors the `OptimizedPortfolioCreator` in [[trading]]:

```yaml
# configs/factor/portfolio.yaml
objective: max_alpha          # max_alpha | min_var | max_ir
constraints:
  long_only: true
  industry_neutral: true
  factor_exposure:
    max_abs: 0.5              # max Barra factor exposure
  turnover:
    max: 0.3                  # max one-way turnover per rebalance
  weight:
    min: 0.0
    max: 0.05                 # max 5% per stock
solver: cvxpy
```

---

## Data Flow

```
configs/factor/{factor_name}.yaml
        ↓
FactorCalculator.from_config(cfg)
        ↓
FactorCalculator.calculate(data)
  → DataAPI / DateDataAccess singletons (TRADE, BS, ...)
  → src/func/ tensor utilities (TsRoller, cs_rank, ...)
        ↓
Raw DataBlock (N_secid × N_date × 1 × 1)
        ↓
StockFactor normalization pipeline
  → fillna → winsorize → neutralize → zscore
        ↓
Normalized DataBlock → PATH.factor/{name}/{date}.feather
        ↓
FactorTestAPI — IC / quantile / long-short analysis
```

---

## Common Patterns / Gotchas

- Factor names must be unique across all categories — the metaclass registry is flat
- `FactorCalculator.calculate()` should be **pure** (no side effects) — caching and saving are handled by `FactorAPI.update()`
- Normalization is applied **after** calculation and is **not** part of `calculate()` — raw and normalized values are both stored
- Financial statement factors need `BS.qtr()` / `BS.ttm()` for point-in-time correctness — never use period-end raw values
- `dependencies` are auto-loaded before `calculate()` is called — list them to avoid manual loading
- `FactorTestAPI` results are written to `PATH.factor / 'tests/'` — check there before re-running expensive tests
