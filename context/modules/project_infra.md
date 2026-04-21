# Project Infrastructure
**Purpose:** Machine config, path management, trading calendar, database I/O, logging, singletons, and numerical utility functions.
**Key source paths:** `src/proj/`, `src/func/`
**Depends on:** nothing (bottom of the dependency stack)

---

## 1. Project Environment — `src/proj/env/`

Import all four top-level objects from the package:
```python
from src.proj.env import MACHINE, PATH, Const, Proj
```

---

### `MACHINE` — `src/proj/env/machine.py`

Static class (class-level attributes only, never instantiated). Auto-resolved at import time from the host name and the `.secret/machines.yaml` config.

**Key attributes:**
```python
MACHINE.name            # socket hostname (first segment)
MACHINE.system_name     # 'linux' | 'windows' | 'macos'
MACHINE.main_path       # project root Path (from pyproject.toml discovery)
MACHINE.config          # ConfFileLazyLoader — reads configs/ YAMLs by stem key
MACHINE.secret          # ConfFileLazyLoader — reads .secret/ YAMLs

MACHINE.cuda_server     # bool: is this a dedicated GPU server
MACHINE.python_path     # interpreter path
MACHINE.share_folder    # Path | None: optional shared network folder
MACHINE.mosek_lic_path  # Path | None
MACHINE.updatable       # bool: may run live data updates
MACHINE.emailable       # bool: may send notification emails
MACHINE.nickname        # display name

MACHINE.belong_to_hfm   # bool: host name starts with hno/hpo
MACHINE.hfm_factor_dir  # Path | None: HFM shared alpha dir (HFM machines only)

MACHINE.platform_server # bool: name == 'mengkjin-server'
MACHINE.platform_coding # bool: is_macos

MACHINE.cpu_count       # os.cpu_count()
MACHINE.max_workers     # 40 on server, cpu_count elsewhere
MACHINE.best_device     # 'CUDA:<name>' | 'MPS' | 'CPU'
MACHINE.timezone        # local timezone object
MACHINE.utc8            # bool: timezone == Asia/Shanghai
```

**`ConfFileLazyLoader`** — used by `MACHINE.config` and `MACHINE.secret`:
```python
# Lazy-loads YAML/JSON from a root directory by slash-separated key path
MACHINE.config('constant/project')                     # returns full dict
MACHINE.config('constant/project', 'vb', default=1)   # nested lookup with fallback
MACHINE.config.get('strategy/model')                   # same as __call__
```
Files are loaded once and cached. Keys may be nested with `/` separators.

**Utility methods:**
```python
MACHINE.info()                          # dict of key attributes for display
MACHINE.machine_main_path('other-host') # Path: main_path of another machine from secrets
```

---

### `PATH` — `src/proj/env/path.py`

Static class of `pathlib.Path` constants. All paths derived from `MACHINE.main_path` — **never hardcode paths**. Called `PATH.mkdir_path()` at import time to ensure all directories exist.

```python
# Project root
PATH.main           # project root
PATH.scpt           # scripts/
PATH.fac_def        # src/res/factor/defs/
PATH.conf           # configs/
PATH.schedule       # configs/model/schedule/

# data/ tree
PATH.data           # data/
PATH.database       # data/DataBase/
PATH.export         # data/Export/
PATH.interim        # data/Interim/
PATH.miscel         # data/Miscellaneous/
PATH.updater        # data/Updater/

# interim sub-paths
PATH.block          # data/Interim/DataBlock/
PATH.batch          # data/Interim/MiniBatch/
PATH.checkpoint     # data/Interim/Checkpoint/
PATH.datacache      # data/Interim/DataCache/
PATH.norm           # data/Interim/HistNorm/

# export sub-paths
PATH.hidden         # data/Export/hidden_feature/
PATH.factor         # data/Export/stock_factor/
PATH.pred           # data/Export/model_prediction/
PATH.fmp            # data/Export/factor_model_port/
PATH.fmp_account    # data/Export/factor_model_account/
PATH.trade_port     # data/Export/trading_portfolio/

# logs/ + results/ + models/
PATH.logs           # logs/
PATH.log_model      # logs/model/
PATH.result         # results/
PATH.rslt_factor    # results/factor/
PATH.rslt_trade     # results/trade/
PATH.model          # models/
PATH.model_nn       # models/nn/
PATH.model_boost    # models/boost/
PATH.model_factor   # models/factor/
PATH.model_st       # models/st/

# resources + templates
PATH.resource       # resources/
PATH.backup         # resources/backup/
PATH.template       # templates/

# machine-local (not shared, not in git)
PATH.local_resources  # .local_resources/
PATH.local_share      # .local_resources/shared/
PATH.local_machine    # .local_resources/<hostname>/
PATH.temp             # .local_resources/temp/
PATH.app_db           # .local_resources/<hostname>/app_db/
PATH.runtime          # .local_resources/<hostname>/runtime/
PATH.optuna           # .local_resources/<hostname>/optuna/
PATH.tensorboard      # .local_resources/<hostname>/tensorboard/
PATH.share_folder     # alias for MACHINE.share_folder
PATH.shared_schedule  # .local_resources/shared/schedule_model/
```

**File I/O helpers:**
```python
PATH.read_yaml(path)                          # → dict (empty {} if missing)
PATH.dump_yaml(data, path, overwrite=False)   # write with indented block style
PATH.read_json(path)                          # → dict
PATH.dump_json(data, path, overwrite=False)

PATH.load_template('css', 'interactive', 'custom')   # → string.Template from .template file
PATH.load_templates('css', 'interactive')            # → dict[stem, Template] for a directory

PATH.file_modified_date(path)   # → int YYYYMMDD (19970101 if missing)
PATH.file_modified_time(path)   # → int YYYYMMDD HHMMSS

PATH.path_at_machine(path, 'other-host')  # translate a local path to another machine's layout
PATH.list_files(directory, fullname=False, recur=False)  # list files, filtering dotfiles
PATH.filter_paths(paths)                  # drop names starting with '.' or '~'
PATH.copytree(src, dst)
PATH.copyfiles(src, dst, bases)
PATH.deltrees(dir, bases)
PATH.mkdir_path()               # ensure all Path class attrs exist on disk
```

---

### `Const` — `src/proj/env/constant/`

Aggregate accessor for domain config trees. All sub-objects are `@singleton` instances.

```python
from src.proj.env import Const

Const.Pref        # Preference — project/interactive/logger/shell_opener config dicts
Const.Factor      # FactorConstants — factor sub-configs
Const.Model       # ModelConstants — model resume flags + strategy list
Const.TradingPort # TradingPortConstants — trading / backtest port lists
```

#### `Const.Pref` — `Preference`
Property-based lazy loads of `configs/constant/preference/` YAML files:
```python
Const.Pref.project       # dict: general project preferences
Const.Pref.interactive   # dict: interactive app preferences
Const.Pref.logger        # dict: logger preferences
Const.Pref.shell_opener  # dict: shell opener preferences
```

#### `Const.Factor` — `FactorConstants`
```python
Const.Factor.UPDATE    # FactorUpdateConfig: start, end, step, init_date, target_dates
Const.Factor.RISK      # RiskModelConfig: market, style, indus, common factor lists
Const.Factor.BENCH     # BenchmarksConfig: availables, defaults, tests, categories, none
Const.Factor.TRADE     # TradeConfig: default(0.00035), harvest(0.002), yale(0.00035)
Const.Factor.ROUNDING  # RoundingConfig: weight(6), exposure(6), ret/turnover(8)
Const.Factor.OPTIM     # PortfolioOptimizationConfig: default, custom (from YAML)
Const.Factor.STOCK     # StockFactorDefinitionConfig: category taxonomy + validators
Const.Factor.FMP       # FMPConfig: creator dict

# Stock factor taxonomy helpers
Const.Factor.STOCK.category0          # list of top-level categories
Const.Factor.STOCK.category1          # dict: cat0 → list[cat1] | None
Const.Factor.STOCK.cat0_to_meta(c0)   # 'stock' | 'market' | 'affiliate' | 'pooling'
Const.Factor.STOCK.cat0_to_cat1(c0)   # allowed cat1 list
Const.Factor.STOCK.cat1_to_cat0(c1)   # reverse lookup
Const.Factor.STOCK.validate_categories(c0, c1)  # raises ValueError if invalid
```

#### `Const.Model` — `ModelConstants`
Resume flags read from `configs/constant/default/model.yaml`:
```python
Const.Model.resume_test         # False | 'last_model_date' | 'last_pred_date'
Const.Model.resume_fmp          # False | 'trailing_N' str
Const.Model.resume_fmp_account  # bool
Const.Model.resume_factor_perf  # bool
Const.Model.strategies          # dict: full strategy/model YAML
```

#### `Const.TradingPort` — `TradingPortConstants`
```python
Const.TradingPort.focused_ports   # list[str]: highlighted port names
Const.TradingPort.tracking_ports  # dict[str, dict]: live trading port specs
Const.TradingPort.backtest_ports  # dict[str, dict]: backtest port specs
```

---

### `Proj` — `src/proj/env/proj.py`

Non-instantiable static facade (metaclass `ProjMeta(NoInstanceMeta)`). Runtime state for verbosity, log files, and cross-module instance references.

```python
from src.proj.env import Proj

Proj.vb                 # Verbosity singleton — global verbosity level
Proj.silence            # Silence context manager
Proj.debug_mode         # bool (from configs/constant/project.yaml)
Proj.show_vb_level      # bool
Proj.instances          # InstanceCollection — trainer / account / factor handles
Proj.email_attachments  # UniqueFileList — files to attach on next email send
Proj.exit_files         # UniqueFileList — files to process on clean exit
Proj.version            # str: package __version__
Proj.log_writer         # LogWriterFile descriptor — current log TextIOWrapper | None

Proj.info()             # dict: MACHINE.info() merged with vb + log_writer
Proj.print_info(once_type='script')   # print once per script or per OS process
Proj.print_disk_info()  # show disk usage
```

#### `Verbosity` — `src/proj/env/verbosity.py`

Singleton. Controls how much a function prints. Levels: `always`(-99) < `min`(0) ≤ `vb` ≤ `max`(10) < `never`(99).

```python
vb = Proj.vb    # or: from src.proj.env.verbosity import Verbosity; vb = Verbosity()

vb.vb           # int: current global level (default 1)
vb.vb_level     # int | None: per-context override (set by WithVbLevel)

vb.set_vb(5)              # change global level
vb.ignore(vb_level=3)     # True if 3 > vb (suppress output)
vb.is_max_level           # bool: vb >= max

# Numeric comparison operators delegate to vb.vb
if vb >= 2: ...

# Context managers
with Verbosity.WithVbLevel(3): ...  # temporary vb_level for this block
with Verbosity.WithVB(5): ...       # temporary global vb, restored on exit

# Resolve symbolic levels
vb('max')     # → 10    vb('min') → 0
vb('never')   # → 99    vb('always') → -99
vb(None)      # → current vb
```

#### `InstanceCollection` — `src/proj/env/variable/ins.py`
Lazy descriptors returning cross-module singletons:
```python
Proj.instances.trainer   # BaseTrainer._trainer  (None if not running)
Proj.instances.account   # PortfolioAccount._account
Proj.instances.factor    # StockFactor._factor
Proj.instances.status()  # dict of non-None slots
```

#### `UniqueFileList` — `src/proj/env/variable/files.py`
Thread-safe deduplicated path list:
```python
Proj.email_attachments.append(path)
Proj.email_attachments.extend(p1, p2)
Proj.email_attachments.pop_all()   # returns list and clears
Proj.email_attachments.ban('tmp')  # reject paths containing 'tmp'
Proj.exit_files.insert(0, path)    # insert at front, deduplicating
```


## 2. Trading Calendar — `CALENDAR`

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

## 3. Database I/O

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

## 4. Logging — `Logger`

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

## 5. Singletons and Utility Patterns

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

## 6. Numerical Utilities — `src/func/`

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
