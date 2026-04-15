# Data Pipeline (`src/data/`)

**Purpose:** Raw data acquisition, loading, caching, preprocessing, and DataBlock tensor construction.
All data access goes through `DateDataAccess` singleton classes or the `DATAVENDOR` facade.
**Key source path:** `src/data/`
**Depends on:** [[project_infra]]

---

## 1. Package Overview & Directory Map

```
src/data/
├── util/              # Shared containers and helpers
│   ├── classes/
│   │   ├── data_block.py       # DataBlock, DataBlockNorm
│   │   ├── module_data.py      # ModuleData, SecidFilter, DateFilter
│   │   ├── datacache.py        # DataCache (disk-backed torch cache)
│   │   ├── nd.py               # NdData (n-dim array + coordinates)
│   │   └── special_dataset.py  # SpecialDataSet (dfl2)
│   ├── df_collection.py        # DFCollection, PLDFCollection
│   ├── stock_info.py           # InfoDataAccess → INFO singleton
│   └── transform.py            # secid_adjust, trade_min_*, adjust_precision
├── loader/            # Singleton data access objects
│   ├── access.py               # DateDataAccess (abstract base)
│   ├── trade_data.py           # TradeDataAccess → TRADE
│   ├── financial_data.py       # FDataAccess subclasses → BS/IS/CF/INDI/FINA, FinData
│   ├── analyst.py              # AnalystDataAccess → ANALYST
│   ├── model_data.py           # RiskModelAccess → RISK
│   ├── min_kline.py            # MinKLineAccess → MKLINE
│   ├── exposure.py             # ExposureAccess → EXPO
│   ├── data_vendor.py          # DataVendor → DATAVENDOR (facade)
│   └── loader.py               # BlockLoader, FrameLoader, FactorLoader, FactorCategory1Loader
├── preprocess/        # Preprocessing pipeline
│   ├── core.py                 # PreProcessorMeta, PreProcessorProperty, PreProcessor hierarchy
│   ├── processors.py           # All PrePro_* concrete subclasses + PrePros accessor
│   └── task.py                 # PreProcessorTask (CLI entry point)
├── update/            # Orchestrated data update pipelines
│   ├── updater.py              # CoreDataUpdater, SellsideDataUpdater, CustomDataUpdater
│   ├── custom/                 # ClassicLabelsUpdater, DailyRiskUpdater, MultiKlineUpdater, etc.
│   └── hfm/                   # JSDataUpdater (JinMeng terminal → server pipeline)
├── download/          # External data acquisition
│   ├── tushare/                # TushareDataDownloader + 9 task files
│   ├── sellside/               # SellsideSQLDownloader, SellsideFTPDownloader
│   └── other_source/          # BaoStock 5min, RiceQuant minute bars
└── crawler/           # Web crawlers
    └── announcement/           # AnnouncementAgent (exchange disclosure crawling)
```

---

## 2. Core Data Container — `DataBlock`

**4D tensor:** `(N_secid, N_date, N_inday, N_feature)`

```python
from src.data.util import DataBlock

# Construction
block = DataBlock(values, secid, date, feature)          # direct
block = DataBlock.from_pandas(df)                        # long-format MultiIndex DataFrame
block = DataBlock.from_polars(df)                        # Polars DataFrame
block = DataBlock.load_raw('trade_ts', 'day', start, end) # from DB (with caching)
block = DataBlock.load_dump(category='preprocess', preprocess_key='day', type='fit')

# Axes
block.values   # torch.Tensor, shape (N_secid, N_date, N_inday, N_feature)
block.secid    # np.ndarray[int]   — stock identifiers
block.date     # np.ndarray[int]   — trading dates (yyyyMMdd)
block.feature  # np.ndarray[str]   — feature names
block.inday    # np.arange(N_inday) — intra-day bar indices

# Alignment (returns aligned copy or modifies in-place)
block.align_secid(secid, inplace=False)
block.align_date(date, inplace=False)
block.align_feature(feature, inplace=False)
block.align_secid_date(secid, date, inplace=False)   # faster combined

# Merging
merged = DataBlock.merge([blk1, blk2],
    secid_method='union', date_method='union',
    inday_method='check', feature_method='stack')

# Slicing
values = block.loc(secid=[...], date=[...], feature=[...])  # positional index match
sub    = block.subset(secid=[...], date=[...])               # returns new block

# Transforms
block.adjust_price()       # multiply prices by adjfactor
block.adjust_volume()      # scale volume/amount
block.ffill()              # forward-fill NaN along date axis
block.mask_values({'list_dt': 91})  # blank pre-listing observations
block.fillna(0)

# Persistence
block.save_dump()          # requires flags: category='preprocess'/'raw', etc.
block = DataBlock.load_dump(category='preprocess', preprocess_key='day', type='fit')

# Normalisation
norm = block.hist_norm(key, start=None, end=20161231)   # returns DataBlockNorm
norms = DataBlock.load_preprocess_norms(['day', '15m'])  # load saved norms

# Conversion
df = block.to_dataframe()  # → pd.DataFrame MultiIndex [secid, date]
```

---

## 3. Supporting Containers

### `NdData` (`src/data/util/classes/nd.py`)
N-dimensional wrapper pairing a numpy/torch array with per-axis coordinate arrays.
Used internally as an intermediate representation in `DataBlock.from_pandas`.

### `DFCollection` / `PLDFCollection` (`src/data/util/df_collection.py`)
Thread-safe date-keyed in-memory DataFrame caches.
- `DFCollection`: pandas backend; lazy-merges into a `long_frame` on multi-date access.
- `PLDFCollection`: Polars backend; stores per-date `pl.DataFrame` objects.

### `DataCache` (`src/data/util/classes/datacache.py`)
Disk-backed persistent cache for `ModuleData` aligned blocks.
Saves data as `torch.save` files with a JSON metadata index.

---

## 4. `DateDataAccess` Singletons

All singletons are imported from `src.data.loader` or via `DATAVENDOR`.

### Abstract base: `DateDataAccess` (`loader/access.py`)
Maintains `DFCollection` / `PLDFCollection` caches per data type.
Key method: `get_specific_data(start, end, data_type, field, prev=True, mask=False, pivot=False)` — the standard PIT (point-in-time) query pattern.

### `TRADE` — `TradeDataAccess` (`loader/trade_data.py`)
```python
from src.data.loader import TRADE

TRADE.get_quotes(start, end, field, adj_price=True)   # adjusted OHLCV
TRADE.get_returns(start, end, return_type='close')    # return_type: close/vwap/open/intraday/overnight
TRADE.get_volumes(start, end, volume_type='amount')
TRADE.get_turnovers(start, end, turnover_type='fr')
TRADE.get_mv(start, end, mv_type='circ_mv')
TRADE.get_market_return(start, end)
TRADE.get_market_amount(start, end)
```

### `INFO` — `InfoDataAccess` (`util/stock_info.py`)
```python
from src.data.util import INFO

INFO.get_secid(date)           # listed secids on a date
INFO.get_desc(date)            # listing description DataFrame
INFO.get_st(date)              # ST/suspended stocks
INFO.get_indus(date)           # Tushare L2 industry classification
INFO.mask_list_dt(df)          # blank pre-IPO observations in a DataFrame
INFO.add_indus(df, date)       # join industry onto a DataFrame
```

### `RISK` — `RiskModelAccess` (`loader/model_data.py`)
```python
from src.data.loader import RISK

RISK.get_exp(date)             # factor exposures
RISK.get_res(date)             # residual returns
RISK.get_exret(start, end)     # cumulative residual (alpha) returns
```

### Financial statement singletons (`loader/financial_data.py`)
```python
from src.data.loader import BS, IS, CF, INDI, FINA

# Common interface for BS/IS/CF/INDI
BS.acc(val, date, lastn=1)       # cumulative (YTD) values
IS.qtr(val, date, lastn=1)       # single-quarter values
CF.ttm(val, date, lastn=1)       # trailing-twelve-months
INDI.qoq(val, date, lastn=1)     # quarter-on-quarter growth
IS.yoy(val, date, lastn=1)       # year-on-year growth

# Latest (cross-sectional) variants
BS.acc_latest(val, date)         # → pd.Series indexed by secid
IS.qtr_latest(val, date)

# Expression-based evaluator
from src.data.loader import FinData
fin = FinData('is@revenue@ttm / bs@total_assets@acc')
series = fin.get_latest(date)      # → pd.Series
df     = fin.get_hist(date, lastn=4)  # → MultiIndex (secid, end_date) DataFrame
```

### `ANALYST` — `AnalystDataAccess` (`loader/analyst.py`)
```python
from src.data.loader import ANALYST

ANALYST.get_trailing_reports(date, n_month=3)   # trailing analyst reports
ANALYST.get_val_est(date, year, val)             # consensus estimate for a year
ANALYST.get_val_ftm(date, val)                   # forward-twelve-months estimate
ANALYST.target_price(date)                       # consensus target price
```

### `MKLINE` — `MinKLineAccess` (`loader/min_kline.py`)
```python
from src.data.loader import MKLINE

MKLINE.get_1min(date)            # → pl.DataFrame
MKLINE.get_5min(date)            # → pl.DataFrame
MKLINE.get_kline(date)           # 1min with fallback to 5min
MKLINE.get_inday_corr(date, val1, val2, lag1=0, lag2=0)
```

### `EXPO` — `ExposureAccess` (`loader/exposure.py`)
```python
from src.data.loader import EXPO

EXPO.get_risks(start, end, field)   # microstructure risk features
```

---

## 5. `DATAVENDOR` Facade (`loader/data_vendor.py`)

The primary entry point for cross-source computed data:

```python
from src.data.loader import DATAVENDOR

# Security universe
secid = DATAVENDOR.secid(date)

# DataBlock getters (lazy-loaded, cached internally)
blk = DATAVENDOR.get_quotes_block(dates)            # price-adjusted OHLCV
blk = DATAVENDOR.get_returns_block(start, end)      # close/vwap daily returns
blk = DATAVENDOR.get_risk_exp(dates)                # CNE5 factor exposures

# Forward return labels
blk = DATAVENDOR.nday_fut_ret(secid, date, nday=10, lag=2)

# Market cap and factor exposures
blk = DATAVENDOR.ffmv(secid, date)                  # float market cap
blk = DATAVENDOR.risk_style_exp(secid, date)        # CNE5 style exposures
blk = DATAVENDOR.risk_industry_exp(secid, date)     # CNE5 industry exposures

# Single-date quotes and returns
df  = DATAVENDOR.day_quote(date, price='close')     # (secid, price) DataFrame
df  = DATAVENDOR.get_quote_ret(date0, date1)        # single-period return
df  = DATAVENDOR.get_miscel_ret(df)                 # arbitrary (secid, start, end) returns

# Financial data
s = DATAVENDOR.get_fin_latest('is@revenue@ttm / bs@total_assets@acc', date)
df = DATAVENDOR.get_fin_hist(expr, date, lastn=4)
df = DATAVENDOR.get_fin_yoy(expr, date, lastn=4)
df = DATAVENDOR.get_fin_qoq(expr, date, lastn=4)
```

All underlying singletons are accessible as class attributes:
`DATAVENDOR.TRADE`, `DATAVENDOR.RISK`, `DATAVENDOR.BS`, `DATAVENDOR.IS`,
`DATAVENDOR.CF`, `DATAVENDOR.INDI`, `DATAVENDOR.FINA`, `DATAVENDOR.ANALYST`,
`DATAVENDOR.MKLINE`, `DATAVENDOR.EXPO`, `DATAVENDOR.INFO`.

---

## 6. Loaders (`loader/loader.py`)

```python
from src.data.loader import BlockLoader, FrameLoader, FactorLoader, FactorCategory1Loader

# Load a DB key as a DataBlock
blk = BlockLoader('trade_ts', 'day', feature=['close','volume']).load(start, end)

# Load a DB key as a raw DataFrame
df  = FrameLoader('trade_ts', 'day').load(start, end)

# Load named factors via FactorCalculator
blk = FactorLoader(['factor_A', 'factor_B']).load(start, end)

# Load all factors in a category1 group
blk = FactorCategory1Loader('quality').load(start, end)
```

---

## 7. PreProcessor System (`preprocess/`)

Preprocessors transform raw database blocks into normalised `DataBlock` tensors.

### Auto-registration
Any class named `PrePro_<key>` is registered automatically by `PreProcessorMeta`.

### `PrePros` accessor
```python
from src.data.preprocess import PrePros

PrePros.keys()                          # list of all registered keys
PrePros.get_processor('day', type='fit')  # instantiate a specific processor
```

### Registered processor keys
| Key | Description |
|-----|-------------|
| `y` | Return labels + cross-sectionally neutralised variants |
| `day` | Daily adjusted OHLCV |
| `15m`, `30m`, `60m` | Intraday bars normalised by daily preclose and turnover |
| `week` | 5-day rolling OHLCV reshaped into inday dimension |
| `style`, `indus` | CNE5 risk model factor exposures |
| `quality`, `growth`, `value`, `earning`, `surprise`, `coverage`, `forecast`, `adjustment` | Factor category groups |
| `hf_momentum`, `hf_volatility`, `hf_correlation`, `hf_liquidity` | High-frequency factor groups |
| `momentum`, `volatility`, `correlation`, `liquidity`, `holding`, `trading` | Further factor groups |
| `dfl2` | Dongfang L2 characteristics — rolling time-series z-score |
| `dfl2cs` | Dongfang L2 characteristics — cross-sectional z-score |

### Incremental update via `load_with_extension`
Processors load existing dumps and only compute new date spans, merging with an `EXTENSION_OVERLAY` overlap to avoid edge discontinuities.

---

## 8. `ModuleData` — End-to-End Model Input Loading

```python
from src.data.util import ModuleData

md = ModuleData(
    data_type_list = ['day', '15m'],   # X block keys
    y_labels       = ['stdret10_1'],   # subset of y features to expose
    use_data       = 'fit',            # 'fit' | 'predict' | 'both'
    filter_secid   = None,             # 'random.200', 'csi300', etc.
    filter_date    = None,             # 'yyyyMMdd~yyyyMMdd'
)
md.load()          # loads, extends, aligns, caches, forward-fills all blocks

md.x               # dict[str, DataBlock] — X feature blocks
md.y               # DataBlock — label block (feature-aligned to y_labels)
md.secid           # np.ndarray from the y block
md.date            # np.ndarray from the y block
md.norms           # dict[str, DataBlockNorm]
```

---

## 9. Data Download Layer

### Tushare (`download/tushare/`)
```python
from src.data.download import TushareDataDownloader

TushareDataDownloader.update()           # incremental update of all registered fetchers
TushareDataDownloader.rollback(date)     # rollback to a specific date
```
Fetcher tasks live in `download/tushare/task/t*.py` and are discovered dynamically via `TushareFetcher.load_tasks()`.

### Sell-side SQL (`download/sellside/from_sql.py`)
```python
from src.data.download import SellsideSQLDownloader

SellsideSQLDownloader.update()   # update all factor_settings entries
```
Downloads factor data from multiple sell-side SQL servers in 60-day chunks.

### JinMeng HFM terminal (`update/hfm/`)
```python
from src.data.update.hfm import JSDataUpdater

updater = JSDataUpdater()
updater.fetch_all()               # terminal side: pack R data into .tar archive
JSDataUpdater.unpack_exist_updaters()  # server side: extract archives
JSDataUpdater.transform_datas()        # convert minute-bar zips to DB format
```

---

## 10. Custom Update Layer (`update/custom/`)

```python
from src.data.update import CustomDataUpdater

CustomDataUpdater.update()    # runs all registered BasicCustomUpdater subclasses
```

Registered updaters (auto-discovered via `BasicCustomUpdater.import_updaters()`):

| Updater | Output DB |
|---------|-----------|
| `ClassicLabelsUpdater` | `labels_ts/ret5`, `ret10`, `ret20`, `ret5_lag`, ... |
| `DailyRiskUpdater` | `exposure/daily_risk` |
| `MultiKlineUpdater` | `trade_ts/5day`, `10day`, `20day` |
| `CustomIndexUpdater` | `index_daily_custom/<index_name>` |
| `MarketDailyRiskUpdater` | `market_daily/risk` |
| `WeekRankLoserUpdater` | `exposure/week_rank_loser` |

---

## 11. Data Flow

```
External Sources
    ├── Tushare API         ─→ download/tushare/  ─→ trade_ts, financial_ts, ...
    ├── Sell-side SQL       ─→ download/sellside/ ─→ sellside/
    ├── JinMeng terminal    ─→ update/hfm/        ─→ trade_js, models/
    ├── BaoStock/RiceQuant  ─→ download/other_source/ → trade_ts/
    └── Web crawlers        ─→ crawler/           ─→ announcement/

Raw DB
    └── DateDataAccess singletons (TRADE, BS, IS, CF, INDI, FINA, RISK, ANALYST, MKLINE, EXPO, INFO)
            └── DataBlock.load_raw() / DB.loads()

Custom Derived Data
    ├── CustomDataUpdater → labels_ts, exposure/, index_daily_custom/
    └── PreProcessor (PrePro_*) → block dumps (.mmap) + norm files (.pt)

Model Inputs
    └── ModuleData.load()
            ├── PrePros.get_processor(key).load(dates)  → DataBlock
            └── DataCache (disk)                         → cached aligned blocks
```

---

## 12. Common Patterns & Gotchas

### Point-in-Time (PIT) Queries
`get_specific_data(start, end, data_type, field, prev=True)` loads data from the *previous* trading day and relabels it to the query date. This is the standard way to avoid look-ahead bias.

### Alignment Convention
`DataBlock.blocks_align(blocks)` uses:
- **secid**: intersection across all blocks
- **date**: union, trimmed to the latest `min(date)` across blocks

This means adding a short-history block will trim dates from all other blocks. Be careful when mixing `dfl2` (short history) with `day` (full history).

### Polars vs pandas
- `MKLINE` always returns `pl.DataFrame`.
- `DFCollection` is pandas; `PLDFCollection` is Polars.
- `DataBlock.from_polars(df)` is the Polars construction path.

### Survivorship Bias
`INFO.mask_list_dt(df)` and `DataBlock.mask_values({'list_dt': 91})` blank out observations before listing + 91 days. Always apply to return labels.

### Forward-fill policy
`DataBlock.blocks_ffill(blocks, fillna='guess')` forward-fills factor blocks (non-OHLCV, non-y) by default. Return blocks (`y`) and intraday blocks are not filled.

### `FREQUENT_DBS` caching
`DataBlock.load_raw()` maintains a local `.mmap` dump for `FREQUENT_DBS` (trade_ts.day, trade_ts.day_val, models.tushare_cne5_exp). Reads of 500+ dates use the dump and only fetch missing dates from the DB.
