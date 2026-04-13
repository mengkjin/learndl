# Boosting Models
**Purpose:** Gradient boosting models (LightGBM, XGBoost, CatBoost, AdaBoost) for stock return prediction and factor combination, with optional Optuna hyperparameter search.
**Key source paths:** `src/res/algo/boost/`, `configs/algo/boost/`
**Depends on:** [[data_pipeline]], [[factor_engine]], [[project_infra]]

---

## Core Concepts

Boosting models share the `AlgoModule` interface with NN models, making them interchangeable in the `ModelAPI` training pipeline. The central class hierarchy:

```
BasicBoostModel (abstract)
├── GeneralBoostModel       — direct fit with fixed params
│   ├── Lgbm
│   ├── XgBoost
│   ├── CatBoost
│   └── AdaBoost
└── OptunaBoostModel        — Optuna-guided hyperparam search before final fit
    ├── OptunaLgbm
    ├── OptunaXgBoost
    └── OcatBoost
```

`GeneralBoostModel.AVAILABLE_BOOSTS` maps string names to concrete classes:
```python
{'lgbm': Lgbm, 'xgboost': XgBoost, 'catboost': CatBoost, 'adaboost': AdaBoost}
```

The `boost_type` field in config determines which concrete class is instantiated.

---

## Key Classes

### `BasicBoostModel` (abstract base)
Defines the interface all boosters implement:
- `fit(boost_input)` — train on a `BoostInput`
- `predict(boost_input)` — return a `BoostOutput`
- Abstract `_fit` / `_predict` implemented by subclasses

### `GeneralBoostModel`
Wraps the four concrete boosters. Selects implementation via `boost_type` config field.

| Method | Purpose |
|--------|---------|
| `fit(boost_input)` | Train with fixed hyperparams from config |
| `predict(boost_input)` | Score samples, return `BoostOutput` |
| `feature_importance()` | Return feature importance array (LGBM/XGBoost only) |

### `OptunaBoostModel`
Adds Optuna hyperparameter search before final fit.

| Item | Description |
|------|-------------|
| `DEFAULT_N_TRIALS` | Default number of Optuna trials (class-level constant) |
| `study_create(direction, ...)` | Create an Optuna study with pruner/sampler |
| `optimize(boost_input, n_trials)` | Run Optuna search loop, record best params |
| `trial_suggest_params(trial, boost_type)` | Suggest a param dict for one trial |

**`trial_suggest_params` parameter spaces by booster:**

*LGBM*
```python
{
  'num_leaves':        trial.suggest_int(20, 300),
  'max_depth':         trial.suggest_int(3, 10),
  'min_child_samples': trial.suggest_int(10, 100),
  'learning_rate':     trial.suggest_float(0.01, 0.3, log=True),
  'n_estimators':      trial.suggest_int(50, 500),
  'subsample':         trial.suggest_float(0.5, 1.0),
  'colsample_bytree':  trial.suggest_float(0.5, 1.0),
  'reg_alpha':         trial.suggest_float(1e-8, 10.0, log=True),
  'reg_lambda':        trial.suggest_float(1e-8, 10.0, log=True),
}
```

*XGBoost*
```python
{
  'max_depth':         trial.suggest_int(3, 10),
  'min_child_weight':  trial.suggest_int(1, 10),
  'learning_rate':     trial.suggest_float(0.01, 0.3, log=True),
  'n_estimators':      trial.suggest_int(50, 500),
  'subsample':         trial.suggest_float(0.5, 1.0),
  'colsample_bytree':  trial.suggest_float(0.5, 1.0),
  'gamma':             trial.suggest_float(1e-8, 1.0, log=True),
  'reg_alpha':         trial.suggest_float(1e-8, 10.0, log=True),
  'reg_lambda':        trial.suggest_float(1e-8, 10.0, log=True),
}
```

*CatBoost*
```python
{
  'depth':               trial.suggest_int(4, 10),
  'learning_rate':       trial.suggest_float(0.01, 0.3, log=True),
  'iterations':          trial.suggest_int(50, 500),
  'l2_leaf_reg':         trial.suggest_float(1e-8, 10.0, log=True),
  'bagging_temperature': trial.suggest_float(0.0, 1.0),
}
```

*AdaBoost*
```python
{
  'n_estimators':  trial.suggest_int(50, 500),
  'learning_rate': trial.suggest_float(0.01, 2.0, log=True),
}
```

---

## Data Contracts

### `BoostInput` (dataclass)
```python
@dataclass
class BoostInput:
    x: torch.Tensor    # shape: (n_sample, n_date, n_feature)
    y: torch.Tensor    # shape: (n_sample,) or (n_sample, n_date)
    w: torch.Tensor    # sample weights
    secid: np.ndarray
    date: np.ndarray
```

Constructors:
- `BoostInput.from_dataframe(df, feature_cols, label_col)` — build from a pandas DataFrame
- `BoostInput.from_tensor(x, y, ...)` — direct tensor construction
- `BoostInput.from_numpy(arr, ...)` — from numpy arrays

The 3D `x` tensor is automatically flattened to 2D `(n_sample * n_date, n_feature)` internally before passing to sklearn-compatible booster APIs.

### `BoostOutput` (dataclass)
```python
@dataclass
class BoostOutput:
    pred:   np.ndarray   # predictions (n_sample,)
    secid:  np.ndarray   # security IDs
    date:   np.ndarray   # dates
    finite: np.ndarray   # boolean mask: True where prediction is finite
```

### `BoostWeightMethod` (dataclass)
Controls how sample weights are computed:
```python
@dataclass
class BoostWeightMethod:
    ts_type: str    # time-series weighting: 'linear', 'exp', 'uniform'
    cs_type: str    # cross-sectional weighting: 'rank', 'uniform'
    bm_type: str    # benchmark: 'none', 'index'
```

---

## Configuration

### Boost algo configs (`configs/algo/boost/`)
| File | Booster | Key Fields |
|------|---------|-----------|
| `lgbm.yaml` | LightGBM | `seqlens`, `objective`, `linear_tree`, `learning_rate`, `num_leaves`, `n_estimators` |
| `xgboost.yaml` | XGBoost | `seqlens`, `objective`, `max_depth`, `learning_rate`, `n_estimators` |
| `catboost.yaml` | CatBoost | `seqlens`, `objective`, `depth`, `iterations`, `learning_rate` |
| `ada.yaml` | AdaBoost | `seqlens`, `n_estimators`, `learning_rate` |

Key shared fields:
- `seqlens` — look-back window lengths (same concept as NN configs; tensor is flattened before use)
- `objective` — loss function (e.g., `'regression'`, `'rank'`)
- `linear_tree` — LGBM-specific: fit linear models at each leaf

### Schedule configs (`configs/model/schedule/`)
| Schedule | Architecture |
|----------|-------------|
| `lgbm_of_factors.yaml` | LightGBM on pre-computed factor inputs |
| `xgb_of_factors.yaml` | XGBoost on pre-computed factor inputs |
| `xgb_of_factors_long.yaml` | XGBoost long-horizon variant |

A schedule config sets `model.module: boost` and `model.algo: lgbm` (or `xgboost`, etc.) to dispatch through `AlgoModule`.

---

## Integration with ModelAPI

Boost models enter through the same `ModelAPI` / `AlgoModule` interface as NN models:
```python
from src.api.model import ModelAPI
ModelAPI.train_model('lgbm_of_factors')
```

`AlgoModule` (`src/res/algo/api.py`) dispatches to boost vs. NN based on `model.module` config field.

Training for boosters is a single `fit` call per CV fold (no epochs/batches). Predictions are written to `PATH.prediction / schedule_name /` in the same feather format as NN outputs.

---

## Common Patterns / Gotchas

- Run factor update (`scripts/2_factor/0_update_factors.py`) before training `lgbm_of_factors` — it requires pre-computed factor values
- Feature importance via `GeneralBoostModel.feature_importance()` is only available for LGBM and XGBoost (not CatBoost/AdaBoost)
- Optuna search can be slow — `DEFAULT_N_TRIALS` controls the budget; reduce for quick experiments
- The 3D `BoostInput.x` is flattened to 2D internally — callers do not need to reshape
- Boost models train much faster than NN models; useful for rapid signal prototyping and baseline comparisons
