# Data Pipeline Review — Pending Modifications

Generated from code review of `src/data/` (2026-04-14).
All items are **do-not-modify-yet** — fix only after confirming the change with the team.

---

## Priority 1: Correctness Bugs

These issues produce silently wrong results or silently do nothing.

### [C1] `from_ftp.py` — `SellsideFTPDownloader.update()` is a dead no-op

**File:** `src/data/download/sellside/from_ftp.py:42-44`

```python
def update(cls):
    return                                       # exits here
    Logger.note(f'Download: {cls.__name__}...')  # unreachable
```

`SellsideDataUpdater.update()` calls this but it silently does nothing.

**Fix:** Remove the premature `return` and implement the FTP update logic, or raise `NotImplementedError` to make the gap visible.

**Potential benefit:** Sell-side FTP data will actually be downloaded.

---

### [C2] `custom_index.py:153` — `total_dates == 0` always False

**File:** `src/data/update/custom/custom_index.py`

```python
if total_dates == 0:   # BUG: total_dates is a list
```

`total_dates` is initialised as `[]` and extended via `.extend()`.
Comparing a list to integer `0` always evaluates to `False` in Python 3.

**Fix:** `if len(total_dates) == 0:` or `if not total_dates:`

**Potential benefit:** The "nothing to update" log branch now triggers correctly.

---

### [C3] `data_block.py` — unreachable guard after assert in `load_dict`

**File:** `src/data/util/classes/data_block.py` (function `load_dict`)

```python
assert file_path.exists() and file_path.is_file()
if not file_path.exists():   # unreachable — assert above already fired
    return {}
```

**Fix:** Remove the dead `if` guard.

**Potential benefit:** Eliminates misleading dead code.

---

### [C4] `processors.py` — Polars z-score expression parenthesisation bug

**Files:**
- `src/data/preprocess/processors.py:215` (`PrePro_dfl2.pre_process`)
- `src/data/preprocess/processors.py:244` (`PrePro_dfl2cs.pre_process`)

**Bug:** `.alias(feat)` is applied to the rolling-std *denominator* expression before division, not to the final quotient. Additionally, `+ 1e-6` is added to the *numerator* (shifting the mean) instead of the denominator (stabilising division).

Current (incorrect):
```python
((pl.col(feat) - pl.col(feat).rolling_mean(...).over("secid")) /
 (pl.col(feat).rolling_std(...).over("secid")).alias(feat) + 1e-6)
```

Correct:
```python
((pl.col(feat) - pl.col(feat).rolling_mean(...).over("secid")) /
 (pl.col(feat).rolling_std(...).over("secid") + 1e-6)).alias(feat)
```

**Potential benefit:** `dfl2` and `dfl2cs` preprocessors will produce correctly normalised values.

---

### [C5] `financial_data.py` — `get_ann_calendar` fragile on empty DataFrame

**File:** `src/data/loader/financial_data.py` (`FDataAccess.get_ann_calendar`)

The loop `for i in range(after_days): ann_calendar += v.shift(i)` starts with `ann_calendar = 0` (integer). If `v` is an empty DataFrame, addition with integer 0 silently returns 0, not a DataFrame. The subsequent `assert isinstance(ann_calendar, pd.DataFrame)` will raise.

**Fix:** Initialise `ann_calendar` as `pd.DataFrame(0, index=v.index, columns=v.columns)`.

**Potential benefit:** Correct behaviour when no announcements exist in the trailing window.

---

### [C7] `custom_index.py` — potential `is_emtpy()` typo

**File:** `src/data/update/custom/custom_index.py`

Verify that `prev_port.is_emtpy()` matches the spelling of the method in `Port`. If `Port` defines `is_empty()` (without the typo), this will raise `AttributeError` at runtime.

**Potential benefit:** Prevents silent AttributeError in custom index calculation.

---

### [C8] `df_collection.py:82` — duplicate `df is not None` check

**File:** `src/data/util/df_collection.py` (`_df_collection.add`)

```python
if df is not None and date not in self.dates and df is not None:  # duplicate
```

**Fix:** `if df is not None and date not in self.dates:`

**Potential benefit:** Code clarity; no functional change.

---

### [C9] `df_collection.py` — `gets()` assertion breaks for `max_len == -1`

**File:** `src/data/util/df_collection.py` (`_df_collection.gets`)

```python
assert len(dates) <= self.max_len   # fails when max_len == -1 (unlimited)
```

**Fix:** `assert self.max_len < 0 or len(dates) <= self.max_len`

**Potential benefit:** `DFCollection(max_len=-1)` can be queried for any number of dates.

---

### [C10] `financial_data.py` — pandas groupby rolling multi-index workaround

**File:** `src/data/loader/financial_data.py` (`FDataAccess._get_data_ttm_hist`, lines ~167-168)

```python
if len(df_ttm.index.names) > 2 and df_ttm.index.names[0] == 'secid' ...:
    df_ttm = df_ttm.reset_index(level=0, drop=True)
```

This silently drops data if pandas changes how `groupby().rolling()` returns a multi-index.

**Fix:** Add a pandas version guard or assert the specific pandas version behaviour; document the workaround.

**Potential benefit:** Prevents silent data loss after pandas upgrades.

---

### [C11] `jsfetcher.py` — calendar CSV re-read per call in `trade_Xday`

**File:** `src/data/update/hfm/jsfetcher.py` (`JSFetcher.trade_Xday`)

`cls.basic_info('calendar')` is called on every invocation, re-reading and parsing the calendar CSV each time.

**Fix:** Cache the result as a class attribute (e.g. `cls._calendar_cache`).

**Potential benefit:** Eliminates repeated I/O; faster batch date processing.

---

## Priority 2: Efficiency Improvements

### [E1] `module_data.py` — repeated `DataBlock.merge` on incremental extension

**File:** `src/data/util/classes/module_data.py` (`ModuleData.extend_blocks`)

Each extension span calls `pre_process` then `merge_others`, allocating a new full-size tensor. For large date ranges this creates unnecessary tensor copies.

**Potential fix:** Pre-allocate the full target tensor and write into it directly.

**Potential benefit:** Reduced peak memory and faster incremental updates.

---

### [E2] `df_collection.py` — `DFCollection.to_long_frame()` is O(N concat)

**File:** `src/data/util/df_collection.py`

Every multi-date query triggers a `pd.concat` of all buffered per-date frames. This is O(N) in the number of buffered dates.

**Potential fix:** Maintain the `long_frame` incrementally as frames are added in `add_one_day`.

**Potential benefit:** Amortised O(1) multi-date queries for high-frequency callers.

---

### [E3] `trade_data.py` — `get_returns` loads one extra date for vwap/open

**File:** `src/data/loader/trade_data.py` (`TradeDataAccess.get_returns`)

For `return_type in ['vwap', 'open']`, the code loads `CALENDAR.td(start, -1)` through `end` (one extra date) to compute `pct_change()`. The extra date is then filtered out. The extra DB load is unnecessary.

**Potential fix:** Load exactly `[start, end]` and compute first-bar returns separately.

**Potential benefit:** Reduced data loading volume.

---

### [E4] `data_vendor.py` — duplicate `get_quote_ret` / `get_quote_ret_new`

**File:** `src/data/loader/data_vendor.py`

Both methods exist and achieve the same result via different implementations.
`get_quote_ret_new` has a commented-out debug comparison block.

**Fix:** Remove `get_quote_ret` (the older DataFrame-join version); rename `get_quote_ret_new` to `get_quote_ret`. Remove the commented debug block.

**Potential benefit:** Removes dead code; DataBlock-based version is more consistent with the rest of the codebase.

---

### [E5] `from_sql.py` — Dongfang forced to serial execution (undocumented)

**File:** `src/data/download/sellside/from_sql.py` (`SellsideSQLDownloader.download_period`)

Dongfang is hard-forced to `method = 'forloop'` regardless of `MAX_WORKERS`. This is presumably a workaround for connection limits but is not documented.

**Fix:** Add a comment explaining the reason; consider making it configurable.

**Potential benefit:** Future readers won't waste time trying to parallelise Dongfang downloads.

---

### [E6] `financial_data.py` — cascaded re-reads in TTM/QTR/ACC methods

**File:** `src/data/loader/financial_data.py`

`_get_data_ttm_hist` calls `_get_data_qtr_hist(lastn + 8)` which calls `_get_data_acc_hist(lastn + 12)`. The cascaded reads fetch significantly more data than the final output needs.

**Potential fix:** Share intermediate `acc`/`qtr` results between callers via a parameter rather than recomputing.

**Potential benefit:** Fewer DB reads; faster financial data access for composite metrics.

---

### [E7] `data_block.py` — `DataBlock.from_pandas()` uses slow xarray path

**File:** `src/data/util/classes/data_block.py` (`DataBlock.from_pandas`)

When the MultiIndex is not a complete Cartesian product, the code uses `xr.Dataset.from_dataframe(df)`, which involves a full pandas → xarray → numpy pipeline.

**Potential fix:** Benchmark a direct `pivot_table` + `reshape` approach; replace if faster.

**Potential benefit:** Faster DataBlock construction from financial data (which often has sparse indices).

---

### [E8] `week_rank_loser.py` — loads full return history since 2007

**File:** `src/data/update/custom/week_rank_loser.py` (`calc_week_rank_loser`)

`DATAVENDOR.get_returns_block(20070101, date)` is called on every date update, loading 17+ years of returns into a DataBlock in memory.

**Fix:** Only load the trailing ~500 trading days (≈50 weeks × 10 days buffer). The algorithm only uses 50 weeks of weekly returns.

**Potential benefit:** ~34× reduction in data loaded per call; prevents OOM on machines with limited RAM.

---

## Priority 3: Code Quality

### [Q1] Remove debug comparison block in `data_vendor.py`

**File:** `src/data/loader/data_vendor.py` (`get_quote_ret_new`, lines ~233-238)

Commented-out debug comparison block should be removed before the duplicate method cleanup.

---

### [Q2] Delete or implement `loader/index_weight.py`

**File:** `src/data/loader/index_weight.py`

Currently a single-line placeholder (effectively empty). Either implement the `IndexWeightAccess` singleton or delete the file.

---

### [Q3] Consolidate duplicated `get_inputs()` / `fillinf()` helpers

**Files:**
- `src/data/update/custom/daily_risk.py`
- `src/data/update/custom/market_daily_risk.py`

Both files define identical (or near-identical) `get_inputs()` and `fillinf()` functions.

**Fix:** Move them to `src/data/update/custom/basic.py` or a new `src/data/update/custom/risk_util.py`.

**Potential benefit:** DRY; single place to update if input schemas change.

---

### [Q4] Refactor `PrePro_15m` / `PrePro_30m` / `PrePro_60m` into a parameterised base

**File:** `src/data/preprocess/processors.py`

These three classes have identical `block_loaders` and `process` implementations
differing only in the DB key (`15min`, `30min`, `60min`).

**Fix:**
```python
class IntraDayTradePreProcessor(TradePreProcessor):
    FREQ : str = ''  # '15m', '30m', '60m'
    def block_loaders(self): ...  # use self.FREQ

class PrePro_15m(IntraDayTradePreProcessor): FREQ = '15m'
class PrePro_30m(IntraDayTradePreProcessor): FREQ = '30m'
class PrePro_60m(IntraDayTradePreProcessor): FREQ = '60m'
```

**Potential benefit:** Bug fixes in one place, not three.

---

## Priority 4: Architecture

### [A1] `DataCache.possible_types` — add `'preprocess'`

**File:** `src/data/util/classes/datacache.py`

Currently only `'module_data'` is in `possible_types`. Preprocessed blocks use a separate dump mechanism (`DataBlock.save_dump`) rather than `DataCache`. Unifying them would simplify the persistence layer.

**Impact:** Medium — requires coordinating `PreProcessor.save_dump` with `DataCache`.

---

### [A2] Move `FinData` into its own module

**File:** `src/data/loader/financial_data.py`

`FinData` (expression-based DSL evaluator) is complex enough to warrant its own file
(`src/data/loader/fin_data.py`). The current `financial_data.py` is already >500 lines.

**Impact:** Low (refactor only; no logic change).

---

### [A3] Externalise JinMeng R paths from `jsfetcher.py`

**File:** `src/data/update/hfm/jsfetcher.py`

Windows R paths like `D:/Coding/ChinaShareModel/...` are hard-coded in multiple classmethods.

**Fix:** Move these paths to `MACHINE.configs('hfm', 'r_paths')` or `MACHINE.secret`.

**Impact:** Low; improves portability across Windows machines.

---

### [A4] Add `__init__.py` to `download/tushare/task/`

**File:** `src/data/download/tushare/task/`

Task modules are currently loaded via `importlib.import_module` dynamic discovery in
`TushareFetcher.load_tasks()`. Adding `__init__.py` would allow direct importability
and enable IDE auto-completion.

**Impact:** Very low; no runtime change.
