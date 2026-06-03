# Training Run Workflow
**Purpose:** End-to-end model training: prepare data, train NN/boost/GP model, evaluate, save predictions.

---

## Prerequisites

1. Data up to date: run [[daily_update]] through step 2 (factors)
2. Schedule config exists: `configs/model/schedule/{schedule_name}.yaml`

---

## Quick Start

```python
from src.api.model import ModelAPI

ModelAPI.train_model('gru_ret1d')     # trains GRU on 1-day return
ModelAPI.train_model('lgbm_of_factors')  # trains LightGBM on factor inputs
```

Or via script:
```bash
python scripts/3_train/0_train.py --schedule gru_ret1d
```

---

## What Happens Internally

1. `ModelConfig.load(schedule_name)` — parse YAML config
2. `DataModule.setup()` — assemble `DataBlock`, split CV folds
3. Walk-forward CV loop (`cv.n_splits` folds):
   - Train on historical window → validate on held-out year
   - Save best checkpoint to `PATH.model/{schedule}/{fold}/`
4. SWA weight averaging (if `swa.enabled: true`)
5. Generate out-of-sample predictions for all dates → `PATH.prediction/{schedule}/`

See [[nn_models]] for full training lifecycle detail.

---

## Monitoring

```bash
python tsboard.py --schedule gru_ret1d    # TensorBoard-style monitor
```

Training metrics (loss, val IC) are logged to `PATH.log/`.

---

## After Training

```python
from src.api.model import ModelAPI
from src.api.summary import SummaryAPI

# Check model performance
SummaryAPI.model_account_summary('gru_ret1d')

# Run backtest with the new predictions
from src.api.trading import TradingAPI
TradingAPI.Backtest('top50_gru')
```

---

## GP Training

GP training has a separate entry point:
```bash
python src/res/gp/main.py   # or the equivalent script in scripts/3_train/
```

See [[gp_strategy]] for GP-specific config and flow.
