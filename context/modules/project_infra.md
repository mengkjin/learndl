# Project Infrastructure
**Purpose:** Machine config, path management, trading calendar, database I/O, logging, singletons, and numerical utility functions.
**Key source paths:** `src/proj/`, `src/func/`
**Depends on:** nothing (bottom of the dependency stack)

---

## 1. Machine Configuration — `MACHINE`

Singleton object loaded from `configs/proj/machine.yaml`. Read via `from src.proj.config import MACHINE`.

Key attributes (≈20 total):
| Attribute | Type | Description |
|-----------|------|-------------|
| `MACHINE.name` | str | Hostname identifier |
| `MACHINE.data_root` | Path | Root directory for all data files |
| `MACHINE.n_jobs` | int | Default parallelism level |
| `MACHINE.cuda` | bool | Whether CUDA is available |
| `MACHINE.device` | str | `'cuda'` or `'cpu'` |
| `MACHINE.ram_gb` | float | Available RAM in GB |
| `MACHINE.storage_type` | str | `'ssd'` or `'hdd'` — affects I/O strategy |

---

## 2. Path Management — `PATH`

Static class with 50+ named paths. All paths are derived from `MACHINE.data_root` — **never hardcode paths**.

```python
from src.proj.path import PATH

PATH.data          # data root
PATH.factor        # factor value store
PATH.model         # trained model checkpoints
PATH.prediction    # model prediction outputs
PATH.portfolio     # portfolio weight files
PATH.log           # log files
PATH.config        # configs root
# ... 40+ more
```

All return `pathlib.Path` objects. Use `PATH.xxx / 'subdir' / 'file.parquet'` for construction.

---

## 3. Constants — `CONST`, `Proj`, 

| Object  | Source                  | Purpose                                                          |
| ------- | ----------------------- | ---------------------------------------------------------------- |
| `CONST` | `src/proj/env/constant` | Global constants (universe lists, index codes, preference, etc.) |
| `Proj`  | `src/proj/env/variable` | Global variables (verbosity, instance registry, etc.)            |


---

## 4. Trading Calendar — `CALENDAR`

Static class with 50+ methods. Import: `from src.proj.calendar import CALENDAR`.

All methods accept dates as `int` (YYYYMMDD), `str`, `datetime`, or `pd.Timestamp`.

### Key method signatures

**Date arithmetic:**
```python
CALENDAR.td(date, offset=0)           # trading day at offset (0=same, 1=next, -1=prev)
CALENDAR.tds(start, end)              # list of trading days in [start, end]
CALENDAR.td_count(start, end)         # number of trading days between two dates
CALENDAR.is_trading_day(date)         # bool
CALENDAR.last_td(date)                # most recent trading day ≤ date
CALENDAR.next_td(date)                # nearest trading day ≥ date
```

**Period endpoints:**
```python
CALENDAR.month_end(date)              # last trading day of the month containing date
CALENDAR.month_start(date)            # first trading day of the month
CALENDAR.quarter_end(date)            # last trading day of the quarter
CALENDAR.year_end(date)               # last trading day of the year
CALENDAR.period_ends(start, end, freq) # list of period-end dates at freq ('M','Q','Y','W')
```

**Report / announcement dates:**
```python
CALENDAR.report_dates()               # standard report disclosure dates for A-shares
CALENDAR.ann_date(period)             # expected announcement date for a report period
```

**Today / recent:**
```python
CALENDAR.today()                      # today as int YYYYMMDD
CALENDAR.last_trading_day()           # most recent past trading day
CALENDAR.latest_date()                # latest date with data available
```

### `TradeDate` / `Dates`
Helper dataclasses wrapping a sorted array of trading dates with slicing support:
```python
dates = TradeDate(start=20200101, end=20231231)
dates.array    # np.ndarray of YYYYMMDD ints
dates[i]       # i-th date
dates.loc(d)   # index of date d
```

---

## 5. Database I/O

### `DBPath` — path resolution for data files
```python
from src.proj.db import DBPath
p = DBPath('trade', 'close', date=20230101)   # resolves to PATH.data / 'trade/close/20230101.feather'
```

### `df_io` — DataFrame read/write
```python
from src.proj.db import df_io
df_io.save(df, path)      # write feather/parquet based on extension
df_io.load(path)          # read feather/parquet, returns pd.DataFrame
df_io.exists(path)        # bool
```

### `ArrayMemoryMap` — memory-mapped numpy arrays
For large arrays that shouldn't be fully loaded into RAM:
```python
from src.proj.db import ArrayMemoryMap
arr = ArrayMemoryMap(path, shape, dtype)
arr[i:j]    # lazy slice — only reads requested rows from disk
```

---

## 6. Logging — `Logger`

Context manager and decorator-based logging. Import: `from src.proj.logger import Logger`.

```python
# Context manager
with Logger('my_task') as log:
    log.info('starting')
    log.warning('something odd')
    log.error('failed')

# Decorator
@Logger.log_calls
def my_function():
    ...
```

Log files written to `PATH.log / '{name}_{date}.log'`.

Additional context managers in `src/proj/logger.py`:
- `Silence` — suppress all stdout/stderr within block
- `Duration` — measure and log elapsed time
  ```python
  with Duration('factor calculation') as d:
      ...
  print(d.elapsed)   # seconds
  ```

---

## 7. Singletons and Utility Patterns

### `SingletonMeta` — thread-safe singleton metaclass
```python
from src.proj.utils import SingletonMeta

class MyService(metaclass=SingletonMeta):
    def __init__(self): ...

# MyService() always returns the same instance
```

### `NoInstanceMeta` — static-only class enforcement
Raises `TypeError` if instantiation is attempted. Used for `CALENDAR`, `PATH`, `MACHINE`.

### `@singleton` — decorator variant
```python
from src.proj.utils import singleton

@singleton
def get_config():
    return load_expensive_config()
```

### `Once` — run-once guard
```python
from src.proj.utils import Once

guard = Once()
if guard.first():
    expensive_setup()
```

### `Device` — device management
```python
from src.proj.utils import Device

Device.get()           # returns torch.device based on MACHINE.cuda
Device.to(tensor)      # move tensor to correct device
Device.empty_cache()   # torch.cuda.empty_cache() if CUDA
```

### `ErrorHandler` — structured exception handling
```python
from src.proj.utils import ErrorHandler

with ErrorHandler(reraise=False) as eh:
    risky_operation()
if eh.caught:
    log.warning(f'error: {eh.exception}')
```

### `FlattenDict` — nested dict → flat dict
```python
from src.proj.utils import FlattenDict
flat = FlattenDict({'a': {'b': 1, 'c': 2}})
# → {'a.b': 1, 'a.c': 2}
```

---

## 8. Numerical Utilities — `src/func/`

Infrastructure-level vectorized operations for numpy/tensor/pandas. No business logic.

### `src/func/basic.py` — index operations
Common index-level operations on arrays and DataFrames:
```python
align_index(df1, df2)          # align two DataFrames to common index/columns
reindex_like(arr, src_idx, tgt_idx)  # reindex array from src to tgt index
isin_mask(arr, values)         # fast boolean mask: arr elements in values set
```

### `src/func/tensor.py` — tensor operations (100+ functions)

The main numerical workhorse. All functions operate on `torch.Tensor` unless noted.

#### `TsRoller` — rolling window engine
```python
from src.func.tensor import TsRoller

roller = TsRoller(window=20, min_periods=5)
roller.mean(x)        # rolling mean along time axis
roller.std(x)         # rolling std
roller.sum(x)         # rolling sum
roller.rank(x)        # rolling rank (returns [0,1])
roller.corr(x, y)     # rolling correlation between two tensors
roller.beta(x, y)     # rolling OLS beta
roller.apply(fn, x)   # rolling apply arbitrary function
```

All `TsRoller` methods preserve the input shape — output tensor has same `(N_secid, N_date, ...)` layout with NaN/0 for initial periods.

#### Selected standalone functions
```python
# Cross-sectional
cs_rank(x)            # cross-sectional rank normalized to [0,1]
cs_zscore(x)          # cross-sectional z-score
cs_demean(x)          # subtract cross-sectional mean

# Time-series
ts_delay(x, n)        # lag by n periods
ts_delta(x, n)        # x - ts_delay(x, n)
ts_pct_change(x, n)   # ts_delta / ts_delay

# Masking / filling
nan_fill(x, method)   # fill NaN: 'ffill', 'mean', 'zero'
nan_mask(x)           # bool tensor: True where NaN
finite_mask(x)        # True where finite (not NaN/Inf)

# Shape utilities
batch_apply(fn, x, batch_size)  # apply fn in batches to avoid OOM
```

### `src/func/transform.py` — statistical transforms
```python
winsorize(x, pct=0.01)          # clip at [pct, 1-pct] quantiles
winsorize_std(x, n_std=3.0)     # clip beyond n_std standard deviations
ols(y, X)                       # OLS: returns (beta, resid, fitted)
cov(x, min_periods=20)          # rolling covariance matrix
neutralize(x, factors)          # OLS-residualize x against factors (cross-sectional)
```

### `src/func/metric.py` — alpha evaluation metrics
```python
ic(pred, target)                # Pearson IC (cross-sectional)
rank_ic(pred, target)           # Spearman rank IC
ic_series(pred_df, target_df)   # IC time series → pd.Series
icir(ic_series)                 # IC Information Ratio
```

### `src/func/linalg.py` — linear algebra
```python
symmetric_orth(A)               # symmetric orthogonalization of matrix A
                                # used for risk-model factor orthogonalization
```
