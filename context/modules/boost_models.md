# Boosting Models
**Purpose:** Gradient boosting models (LightGBM, XGBoost, CatBoost, AdaBoost) for stock return prediction and factor combination, with optional Optuna hyperparameter search.
**Key source paths:** `src/res/algo/boost/`, `configs/algo/boost/`
**Depends on:** [[data_pipeline]], [[factor_engine]], [[project_infra]]

---

## Core Concepts

Boosting models share the `AlgoModule` interface with NN models, making them interchangeable in the `ModelAPI` training pipeline. The central class hierarchy:

```
BasicBoostModel (abstract)           — src/res/algo/boost/util/basic.py
└── [concrete back-ends]             — src/res/algo/boost/booster/
    ├── Lgbm                         — lgbm.py
    ├── XgBoost                      — xgboost.py
    ├── CatBoost                     — catboost.py
    └── AdaBoost                     — ada.py

GeneralBoostModel                    — booster/general.py
    Wraps one of the concrete back-ends selected by boost_type.

OptunaBoostModel(GeneralBoostModel)  — booster/optuna.py
    Runs an Optuna study to pick best params, then delegates to GeneralBoostModel.
```

`AVAILABLE_BOOSTS` in `booster/general.py` maps string names to concrete classes:
```python
{'lgbm': Lgbm, 'ada': AdaBoost, 'xgboost': XgBoost, 'catboost': CatBoost}
```

The `boost_type` argument to `GeneralBoostModel` / `AlgoModule.get_boost()` selects the back-end.

---

## Key Classes

### `BasicBoostModel` (abstract base) — `boost/util/basic.py`
Defines the interface all concrete boosters implement:
- `fit(train, valid, silent)` — train; sub-classes call `boost_fit_inputs()` to normalise inputs
- `predict(x)` — return a `BoostOutput`
- `to_dict() / load_dict()` — serialisation (each sub-class stores its own native model)
- `import_data(train, valid, test)` — loads and converts to `BoostInput` via `to_boost_input()`
- `df_input(factor_data, idx, windows_len)` — walk-forward split returning `{train, valid, test}` dicts

Parameter flow:
- `DEFAULT_TRAIN_PARAM` keys are the only valid `train_param` keys (validated in `assert_param`).
- `n_bins` in `train_param` triggers categorical label conversion; automatically capped at
  `DEFAULT_CATEGORICAL_MAX_BINS` (10) for softmax objectives.

### `GeneralBoostModel` — `booster/general.py`
Thin dispatcher that splits a flat `params` dict into three namespaces:
- Keys in `BoostWeightMethod.__slots__` → `weight_param`
- Key `'verbosity'` → `fit_verbosity` (controls log frequency)
- Everything else → `train_param`

Calling the instance (`model(x)`) wraps `x` into a `BoostInput` and calls `forward()`.

### `OptunaBoostModel` — `booster/optuna.py`
Adds Optuna HPO to the fit pipeline:

| Method | Description |
|--------|-------------|
| `study_create(direction, silent)` | Create named study, persisted to SQLite by default |
| `study_optimize(n_trials, silent)` | Run trials; each calls `study_objective()` |
| `study_objective(trial)` | Suggests params, fits, returns mean RankIC on valid set |
| `trial_suggest_params(trial)` | Per-booster Optuna search space |
| `study_plot(plot_type, ...)` | Visualise study results |

Trial limits: LGBM/XGBoost/CatBoost capped at 100 trials; AdaBoost at 20.

**`trial_suggest_params` search spaces (actual implementation):**

*LGBM*
```python
{
  'objective':        ['mse', 'mae'],
  'learning_rate':    log-uniform [1e-3, 0.3],
  'max_depth':        int [3, 12],
  'num_leaves':       int [20, 100] step 10,
  'min_data_in_leaf': int [10, 50] step 10,
  'reg_alpha':        log-uniform [1e-7, 100],
  'reg_lambda':       log-uniform [1e-6, 100],
  'feature_fraction': [0.5, 1.0] step 0.1,
  'bagging_fraction': [0.5, 1.0] step 0.1,
}
```

*XGBoost*
```python
{
  'objective':        ['reg:squarederror', 'reg:absoluteerror'],
  'learning_rate':    log-uniform [1e-3, 0.3],
  'max_depth':        int [3, 12],
  'subsample':        [0.5, 1.0] step 0.1,
  'colsample_bytree': [0.5, 1.0] step 0.1,
  'reg_alpha':        log-uniform [1e-7, 100],
  'reg_lambda':       log-uniform [1e-6, 100],
}
```

*CatBoost*
```python
{
  'objective':            ['RMSE', 'MAE'],
  'learning_rate':        log-uniform [1e-3, 0.3],
  'max_depth':            int [3, 12],
  'l2_leaf_reg':          log-uniform [1e-3, 10.0],
  'bagging_temperature':  [0.0, 1.0],
  'random_strength':      log-uniform [1e-9, 10.0],
  'od_type':              ['IncToDec', 'Iter'],
  'min_data_in_leaf':     int [1, 100],
}
```

*AdaBoost*
```python
{
  'n_learner':     int [10, 50] step 5,
  'n_bins':        int [10, 30] step 5,
  'max_nan_ratio': [0.5, 0.9] step 0.1,
}
```

---

## Data Contracts

### `BoostInput` — `boost/util/boost_io.py`
```python
@dataclass
class BoostInput:
    x:            torch.Tensor       # (n_sample, n_date, n_feature) — 3-D
    y:            torch.Tensor|None  # (n_sample, n_date) — 2-D
    w:            torch.Tensor|None  # (n_sample, n_date); None → computed via weight_method
    secid:        np.ndarray         # (n_sample,)
    date:         np.ndarray         # (n_date,)
    feature:      np.ndarray         # (n_feature,)
    weight_param: dict               # forwarded to BoostWeightMethod
    n_bins:       int|None           # when set, y is converted to int category labels
```

**Flat accessor methods** (NaN rows dropped, date-major order by default):
- `X()` → `(n_finite, n_use_feature)` tensor
- `Y()` → `(n_finite,)` tensor (categorical int or float)
- `W()` → `(n_finite,)` weight tensor
- `SECID()` / `DATE()` → flat index arrays

Constructors:
- `BoostInput.from_dataframe(df, weight_param)` — last column is label; secid/date auto-detected
- `BoostInput.from_tensor(x, y, w, secid, date, feature, weight_param)` — handles 2-D and 3-D `x`
- `BoostInput.from_numpy(...)` — thin wrapper around `from_tensor`
- `BoostInput.concat(datas)` — union-merges a list of `BoostInput` objects along all axes

### `BoostOutput` — `boost/util/boost_io.py`
```python
@dataclass
class BoostOutput:
    pred:   torch.Tensor    # flat predictions of length n_finite
    secid:  np.ndarray      # (n_sample,)
    date:   np.ndarray      # (n_date,)
    finite: torch.Tensor    # bool mask (n_sample, n_date) — non-NaN positions
    label:  torch.Tensor    # original continuous y for evaluation
```

`to_2d()` reconstructs the full `(n_sample, n_date)` grid filling non-finite positions with 0.

### `BoostWeightMethod` — `boost/util/boost_io.py`
Three-axis multiplicative weight calculator:

| Axis | Param | Options | Effect |
|------|-------|---------|--------|
| Time-series | `ts_type` | `'lin'`, `'exp'`, `None` | Recent dates get higher weight |
| Cross-sectional | `cs_type` | `'top'`, `'ones'`, `None` | Top-ranked or positive-label up-weighting |
| Benchmark | `bm_type` | `'in'`, `None` | Securities in `bm_secid` get weight ×2 |

Final weight: `w = cs_weight * ts_weight * bm_weight` (element-wise, shape `(n_sample, n_date)`).

### `LgbmPlot` — `booster/lgbm.py`
Visualisation helper accessed via `lgbm_model.plot`:

| Method | Output |
|--------|--------|
| `training()` | Loss curve with best-iteration marker |
| `importance()` | Feature importance bar chart |
| `histogram(feature_idx)` | Split-value histograms |
| `tree(num_trees_list)` | Rendered tree structures |
| `shap(train)` | SHAP summary + per-feature dependence (requires `shap` package) |
| `sdt(train)` | Single-distillation tree |
| `pdp(train)` | Partial dependence plots |

All methods save to `plot_path` when configured; methods that need a path return early if it is `None`.

---

## Custom AdaBoost Implementation

`AdaBoost` in `booster/ada.py` is a fully custom implementation (not sklearn-based):

- **Input transform**: rank-percentile → integer bins `[0, n_bins-1]`; NaN → -1
- **Label transform**: tertile rank → ternary `{-1, 0, +1}`
- **`StrongLearner`**: ensemble of `WeakLearner` stumps; weights updated via `exp(-y * y_pred)`
- **`WeakLearner`**: single decision stump selected by minimum Gini impurity (weighted `sqrt(pos*neg)` sum); stores per-bin log-odds predictions
- **Predict**: mean of stump scores, z-scored by std

Training combines train+valid before fitting (no early stopping).  Only `cs_type in ['ones', None]` is supported for weight computation.

---

## Configuration

### Boost algo configs (`configs/algo/boost/`)
| File | Booster | Key Fields |
|------|---------|-----------|
| `lgbm.yaml` | LightGBM | `objective`, `linear_tree`, `learning_rate`, `num_leaves`, `num_boost_round` |
| `xgboost.yaml` | XGBoost | `objective`, `max_depth`, `learning_rate`, `num_boost_round` |
| `catboost.yaml` | CatBoost | `objective`, `max_depth`, `learning_rate`, `num_boost_round` |
| `ada.yaml` | AdaBoost | `n_learner`, `n_bins`, `max_nan_ratio` |

### Schedule configs (`configs/model/schedule/`)
A schedule config sets `model.module: boost` and passes `boost_type` to dispatch
through `AlgoModule.get_boost()`.

---

## Integration with ModelAPI

```python
from src.api.model import ModelAPI
ModelAPI.train_model('lgbm_of_factors')
```

`AlgoModule` (`src/res/algo/api.py`) dispatches to boost vs. NN based on
`model.module` config field.  Training for boosters is a single `fit` call per
CV fold (no epochs/batches).  Predictions are written to
`PATH.prediction / schedule_name /` in the same feather format as NN outputs.

---

## Common Patterns / Gotchas

- Run factor update before training `lgbm_of_factors` — it requires pre-computed factor values
- Feature importance is only available for LGBM (`lgbm_model.plot.importance()`) and implicitly for XGBoost
- Optuna search can be slow — `DEFAULT_N_TRIALS` controls the budget; reduce for quick experiments
- `BoostInput.from_dataframe` treats the **last column** as the label — column order matters
- `eval_metric: None` must be popped before CatBoost `fit()`; this is done inside `CatBoost.fit()` automatically
- XGBoost serialisation uses a temporary JSON file (no `model_to_string` API in XGBoost)
- `AlgoModule.export_available_modules()` is called at import time and writes a file to `PATH.temp` — a known side-effect (see `TODO_res_algo.md`)
