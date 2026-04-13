# System Overview

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     scripts/ (pipeline)                          │
│  0_check  1_autorun  2_data  3_factor  4_train  5_test          │
│           6_predict  7_trading                                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ calls
┌───────────────────────▼─────────────────────────────────────────┐
│                  src/api/  (API facade)                          │
│   DataAPI  FactorAPI  ModelAPI  TradingAPI  UpdateAPI            │
└──────┬────────────┬────────────┬──────────────┬─────────────────┘
       │            │            │              │
┌──────▼──┐  ┌──────▼──────┐  ┌─▼────────────┐  ┌──────▼──────────┐
│src/data/│  │ src/res/    │  │ src/res/     │  │src/res/trading/ │
│Data     │  │ factor/     │  │ algo/        │  │Portfolio        │
│Pipeline │  │ Factor      │  │ nn/, boost/  │  │Construction     │
│         │  │ Engine      │  │+src/res/model│  │                 │
└──────┬──┘  └──────┬──────┘  └─────────────┘  └─────────────────┘
       │            │
┌──────▼────────────▼────────────────────────────────────────────┐
│           src/proj/ + src/func/  (infrastructure)               │
│   env/  cal/  db/  log/  util/  core/                          │
│   basic.py  tensor.py  transform.py  metric.py                 │
└────────────────────────────────────────────────────────────────┘
```

## Layer Table

| Layer | Source Path | Purpose |
|-------|-------------|---------|
| Infrastructure | `src/proj/` | Machine config, paths, calendar, DB I/O, logging, singletons |
| Numerical utils | `src/func/` | Vectorized tensor/numpy/pandas operations, rolling windows |
| Data pipeline | `src/data/` | Raw data download, loading, preprocessing, DataBlock construction |
| Factor engine | `src/res/factor/` | Factor calculation, normalization, testing, risk model |
| Algo modules | `src/res/algo/` | NN architectures (`nn/`) and boosting models (`boost/`) |
| Model framework | `src/res/model/` | Training framework: DataModule, callbacks, model_module |
| GP strategy | `src/res/gp/` | DEAP genetic programming for symbolic factor discovery |
| Trading | `src/res/trading/` | Portfolio construction, live tracking, backtesting |
| API facade | `src/api/` | Stable public interfaces over all research modules |
| Interactive app | `src/interactive/` | Streamlit app (`launch.py`): script runner, task monitor |
| Pipeline scripts | `scripts/` | Numbered end-to-end workflow scripts (0–7) |

## API Facade (`src/api/`)

| File | Class / Functions | Key Methods |
|------|-------------------|-------------|
| `data.py` | `DataAPI` | `get_data()`, `update()`, `preprocess()` |
| `factor.py` | `FactorAPI` | `calculate()`, `update()`, `test()`, `normalize()` |
| `model.py` | `ModelAPI` | `train_model()`, `schedule_model()`, `test_factor()` |
| `trading.py` | `TradingAPI` | `update()`, `Backtest()`, `Analyze()`, `backtest_rebuild()` |
| `summary.py` | `SummaryAPI` | `model_account_summary()`, `backtest_port_account_summary()` |
| `update.py` | `UpdateAPI` | Orchestrated multi-step update pipeline |
| `tsboard.py` | `TSBoardAPI` | TensorBoard-style training monitor |
| `notification.py` | `NotificationAPI` | Email / alert dispatch |

## Pipeline Scripts (`scripts/`)

Scripts starting with `.` are hidden from the interactive app. Scripts in `_miscellaneous/` are also hidden (folder starts with `_`). Visible scripts are discovered by `PathItem.iter_folder`.

| Folder | Purpose | Key Scripts |
|--------|---------|-------------|
| `0_check/` | Maintenance & checks | `test_streamlit`, `update_context` |
| `1_autorun/` | Scheduled pipeline runs | `daily_update`, `weekly_update`, `rollback_update` |
| `2_data/` | Data management | `train_data`, `pack_model_files`, `unpack_model_files` |
| `3_factor/` | Factor compute | `update_factors`, `recalc_one`, `analyze` |
| `4_train/` | Model training | `quick_train`, `train_model`, `schedule_model`, `tensorboard` |
| `5_test/` | Model & factor testing | `test_model`, `test_factor` |
| `6_predict/` | Prediction update | `recalc_preds` (`.0`/`.1` hidden) |
| `7_trading/` | Portfolio ops | `analyze_tracking`, `analyze_backtest`, `rebuild_backtest` (`.0` hidden) |

## Config System (`configs/`)

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `configs/algo/nn/` | Per-architecture hyperparams | `gru.yaml`, `tra.yaml`, `patch_tst.yaml`, `lgbm.yaml`, … |
| `configs/algo/boost/` | Boost model params | `lgbm.yaml`, `xgboost.yaml`, `catboost.yaml`, `ada.yaml` |
| `configs/algo/gp/` | GP evolution params | `params.yaml` |
| `configs/model/` | Training framework config | `model.yaml`, `train.yaml`, `callbacks.yaml`, `input.yaml`, `env.yaml`, `conditional.yaml` |
| `configs/model/schedule/` | Training schedules | `gru_dfl2.yaml`, `lgbm_of_factors.yaml`, `xgb_of_factors.yaml`, … |
| `configs/model/default/` | Default field values | `required.yaml`, `optional.yaml` |
| `configs/setting/` | Project-level settings | `model.yaml`, `trading_port.yaml` |
| `configs/preference/` | App preferences | `project.yaml`, `interactive.yaml`, `logger.yaml`, `shell_opener.yaml` |
| `configs/util/factor/` | Factor utility config | `default_opt_config.yaml`, `risk_factors.yaml` |
| `configs/util/industry/` | Industry classification | `tushare.yaml` |
| `configs/util/transform/` | Preprocessing params | *(transform pipeline configs)* |
| `configs/util/calendar/` | Calendar overrides | *(holiday / trading day overrides)* |

> **Note:** Portfolio configs live at `configs/setting/trading_port.yaml` (not `configs/trading/`).
> Schedule configs live at `configs/model/schedule/` (not `configs/model/schedule_name/`).

## Core Data Container — `DataBlock` / `Stock4D`

4D tensor: **`(N_secid × N_date × N_inday × N_feature)`**

All model inputs, factor outputs, and data pipeline results flow through this shape.

## `src/res/factor/` Subdirectory Map

| Subdir | Purpose |
|--------|---------|
| `calculator/` | `FactorCalculator` abstract base + metaclass registry |
| `defs/` | Concrete factor definitions (market, stock, pooling, affiliate) |
| `analytic/` | IC analysis, factor test framework |
| `fmp/` | Factor model portfolio: generator + optimizer |
| `loader/` | Factor value I/O (load from / save to disk) |
| `risk/` | Risk model (factor covariance, specific risk) |
| `util/` | Shared utilities: agency, classes, plot, stats |

## `src/res/model/` Subdirectory Map

| Subdir | Purpose |
|--------|---------|
| `callback/` | Training callbacks (early stopping, checkpointing, SWA, LR scheduler) |
| `data_module/` | `DataModule` — data loading, batching, cross-validation splits |
| `model_module/` | Core training loop + application layer (predictor, calculator, portfolio) |
| `model_module/module/` | Model state, config, trainer |
| `model_module/application/` | `ModelPredictor`, `ModelExtractor`, `ModelCalculator` |
| `util/` | Shared model utilities |

## Design Patterns

| Pattern | Where Used | Description |
|---------|-----------|-------------|
| Metaclass auto-registration | `FactorCalculator`, `PreProcessor` | Subclasses auto-register by class name on definition |
| Singleton | `DateDataAccess` subclasses, `CALENDAR` | Single instance per type, lazy-initialized |
| `SingletonMeta` | `src/proj/util/` | Thread-safe singleton metaclass |
| Config-driven dispatch | `AlgoModule` | Selects NN vs boost from `model.module` field in schedule config |
| API facade | `src/api/` | Stable public surface over `src/res/` and `src/data/` internals |
| `@ScriptTool` | All `scripts/**/*.py` | Wraps `main()` for Streamlit task runner + email + locking |
| Script header YAML | All `scripts/**/*.py` | `# key: value` comments parsed by `ScriptHeader.read_from_file()` |
| Numbered pipeline | `scripts/` | Explicit ordering: 0_check → 1_autorun → … → 7_trading |
