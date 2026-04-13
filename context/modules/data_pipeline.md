# Data Pipeline
**Purpose:** Raw data loading, caching, preprocessing, and DataBlock construction. All data access goes through `DateDataAccess` singleton classes or `DataAPI`.
**Key source paths:** `src/data/`
**Depends on:** [[project_infra]]

---

## Core Data Container — `DataBlock` / `Stock4D`

4D tensor: **`(N_secid × N_date × N_inday × N_feature)`**

```python
from src.data.block import DataBlock

db = DataBlock(data, secid, date, inday, feature)
db.data        # torch.Tensor, shape (N_secid, N_date, N_inday, N_feature)
db.secid       # np.ndarray of security IDs
db.date        # np.ndarray of YYYYMMDD ints
db.inday       # np.ndarray of intra-day labels
db.feature     # np.ndarray of feature names

# Slicing
db.sel(secid=[...], date=[...])     # select by value
db.iloc(secid=slice(0,100))        # select by position

# Conversion
db.to_dataframe()    # → pd.DataFrame (MultiIndex: secid × date)
db.to_tensor()       # → torch.Tensor
```

`Stock4D` is an alias for `DataBlock`. Both names appear in the codebase.

### `DFCollection`
Thin wrapper around a dict of named DataFrames (used before assembly into a DataBlock):
```python
col = DFCollection()
col['close'] = df_close
col['volume'] = df_volume
block = col.to_datablock()
```

---

## DateDataAccess Singletons

Seven singleton classes, one per data domain. Each caches its data after first load.
All share the base pattern: `CLASS.method(date_range, universe, ...)`.

Import pattern: `from src.data.access import TRADE, ANALYST, ...`

---

### `TRADE` — market trading data (15+ methods)

| Method | Returns | Description |
|--------|---------|-------------|
| `TRADE.get_close(dates, universe)` | `DataBlock` | Adjusted close price |
| `TRADE.get_open(...)` | `DataBlock` | Open price |
| `TRADE.get_high(...)` | `DataBlock` | High price |
| `TRADE.get_low(...)` | `DataBlock` | Low price |
| `TRADE.get_vwap(...)` | `DataBlock` | Volume-weighted average price |
| `TRADE.get_volume(...)` | `DataBlock` | Trading volume |
| `TRADE.get_amount(...)` | `DataBlock` | Trading amount (CNY) |
| `TRADE.get_turnover(...)` | `DataBlock` | Turnover rate |
| `TRADE.get_ret(...)` | `DataBlock` | Daily return |
| `TRADE.get_ret_inday(...)` | `DataBlock` | Intraday return (open→close) |
| `TRADE.get_mktcap(...)` | `DataBlock` | Market capitalization |
| `TRADE.get_free_mktcap(...)` | `DataBlock` | Free-float market cap |
| `TRADE.get_universe(date, name)` | `np.ndarray` | Security IDs in a named universe |
| `TRADE.get_status(dates, universe)` | `DataBlock` | Trading status (suspended/normal) |
| `TRADE.get_adj_factor(...)` | `DataBlock` | Price adjustment factor |

---

### `ANALYST` — analyst consensus estimates

| Method | Returns | Description |
|--------|---------|-------------|
| `ANALYST.get_eps_fwd(dates, n_fwd)` | `DataBlock` | Forward EPS estimate (n_fwd periods ahead) |
| `ANALYST.get_target_price(dates)` | `DataBlock` | Consensus target price |
| `ANALYST.get_rating(dates)` | `DataBlock` | Consensus rating score |
| `ANALYST.get_coverage(dates)` | `DataBlock` | Number of analysts covering |
| `ANALYST.get_revision(dates, window)` | `DataBlock` | EPS revision over window |

---

### `BS` / `IS` / `CF` / `INDI` / `FINA` — financial statements

Five statement singletons with identical method interfaces:

| Singleton | Data Source |
|-----------|-------------|
| `BS` | Balance Sheet |
| `IS` | Income Statement |
| `CF` | Cash Flow Statement |
| `INDI` | Financial Indicators (derived ratios) |
| `FINA` | Combined financial data |

Each exposes four temporal views:
```python
BS.acc(item, dates)    # accumulated (ytd) value
BS.qtr(item, dates)    # single-quarter value
BS.ttm(item, dates)    # trailing twelve months
BS.qoq(item, dates)    # quarter-over-quarter growth
BS.yoy(item, dates)    # year-over-year growth
```

`item` is a string column name (e.g., `'total_assets'`, `'net_profit'`, `'oper_cash_flow'`).

---

### `MKLINE` — market microstructure (Polars-based)

High-frequency and minute-level data. Uses **Polars** (not pandas) internally for performance.

```python
MKLINE.get_minute_bar(dates, universe, freq='1min')   # minute OHLCV
MKLINE.get_inday_ret(dates, universe)                 # intraday return profile
MKLINE.get_auction(dates, universe)                   # opening/closing auction data
```

Returns `DataBlock` with `N_inday > 1`.

---

### `EXPO` — factor exposures

```python
EXPO.get_style(dates, universe)       # Barra-style factor exposures
EXPO.get_industry(dates, universe)    # Industry classification (one-hot or code)
EXPO.get_size(dates, universe)        # Size factor exposure
```

---

### `RISK` — risk model

```python
RISK.get_cov(date)                    # factor covariance matrix (pd.DataFrame)
RISK.get_specific_risk(dates)         # stock-specific risk (idiosyncratic vol)
RISK.get_factor_return(dates)         # factor return time series
```

---

### `INFO` — static security information

```python
INFO.get_industry(date)               # industry classification → pd.Series (secid → industry)
INFO.get_name(secids)                 # Chinese company names
INFO.get_list_date(secids)            # IPO dates
INFO.get_delist_date(secids)          # delisting dates (NaN if still listed)
INFO.get_st_flag(dates)               # ST/PT flag per date
```

---

## `DataVendor` — raw data source (`DATAVENDOR`)

Singleton that wraps external data provider APIs. ~25 methods.

```python
from src.data.vendor import DATAVENDOR

DATAVENDOR.update_trade(start, end)         # pull and save market data
DATAVENDOR.update_financial(start, end)     # pull and save financial statements
DATAVENDOR.update_analyst(start, end)       # pull and save analyst data
DATAVENDOR.update_index(start, end)         # pull index constituent data
DATAVENDOR.available_dates(data_type)       # dates already saved locally
```

Data is saved to `PATH.data / data_type /` via `df_io` from [[project_infra]].

---

## `PreProcessor` — 27 subclasses

Metaclass auto-registration: every subclass of `PreProcessor` is automatically available by name.

```python
from src.data.preprocess import PreProcessor

pp = PreProcessor.get('winsorize_cs')   # get by registered name
result = pp.transform(datablock)
```

### Available processors (selected)

| Name | Operation |
|------|-----------|
| `winsorize_cs` | Cross-sectional winsorization at 1%/99% |
| `winsorize_ts` | Time-series winsorization |
| `zscore_cs` | Cross-sectional z-score normalization |
| `zscore_ts` | Time-series z-score |
| `rank_cs` | Cross-sectional rank → [0,1] |
| `fillna_ffill` | Forward-fill NaN values |
| `fillna_mean` | Fill NaN with cross-sectional mean |
| `neutralize_industry` | OLS-residualize against industry dummies |
| `neutralize_size` | OLS-residualize against log market cap |
| `neutralize_style` | OLS-residualize against style factors |
| `log_transform` | Log1p transform |
| `clip_std` | Clip at ±3 standard deviations |
| `demean_cs` | Subtract cross-sectional mean |
| `standardize` | Mean=0, std=1 (combined zscore) |
| `market_adj` | Subtract market return |
| `industry_adj` | Subtract industry median |

All processors are stateless transforms — they do not store fitted parameters.

---

## `DataAPI` — public interface

```python
from src.api.data import DataAPI

DataAPI.get_data(data_type, dates, universe, **kwargs)   # unified data access
DataAPI.update(data_type, start, end)                    # trigger data refresh
DataAPI.preprocess(datablock, pipeline)                  # apply list of PreProcessors
DataAPI.build_block(features, dates, universe)           # assemble DataBlock from feature list
```

`pipeline` is a list of processor names, e.g. `['fillna_ffill', 'winsorize_cs', 'zscore_cs']`.

---

## Data Flow

```
External vendor APIs
        ↓
DATAVENDOR.update_*()
        ↓
Raw files: PATH.data/{type}/{date}.feather
        ↓
DateDataAccess singletons (TRADE, BS, MKLINE, ...)
  → in-memory cache (lazy-loaded per date range)
        ↓
DataBlock (N_secid × N_date × N_inday × N_feature)
        ↓
PreProcessor pipeline (winsorize → zscore → neutralize → ...)
        ↓
Preprocessed DataBlock → factor engine / model training
```

---

## Common Patterns / Gotchas

- All `DateDataAccess` singletons are lazy — first call loads from disk, subsequent calls return cached data; call `.clear_cache()` to force reload
- `MKLINE` uses Polars internally; its output is converted to `DataBlock` at the boundary — don't pass Polars DataFrames into other modules
- Financial statement data (`BS`, `IS`, `CF`) uses point-in-time logic — `qtr(item, date)` returns the value known **as of** `date`, not the period-end value
- `TRADE.get_universe(date, 'hs300')` returns the HS300 constituents as of that specific date (survivorship-bias-free)
- When building DataBlocks manually, always align `secid` and `date` axes using `reindex_like` from [[project_infra]] `src/func/basic.py`
