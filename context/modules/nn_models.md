# NN Models
**Purpose:** PyTorch-based neural network training, prediction, and evaluation pipeline. 18 architecture implementations with Stochastic Weight Averaging and a shared ModelAPI interface.
**Key source paths:** `src/res/algo/nn/` (architectures), `src/res/model/` (training framework), `src/api/model.py`
**Depends on:** [[data_pipeline]], [[factor_engine]], [[project_infra]]

---

## `ModelAPI` — public interface

```python
from src.api.model import ModelAPI

ModelAPI.train_model(schedule_name)          # train using a schedule config
ModelAPI.Trainer                             # training orchestrator
ModelAPI.Predictor                           # run inference on trained model
ModelAPI.Extractor                           # extract intermediate representations
ModelAPI.Calculator                          # factor-style scoring
ModelAPI.PortfolioBuilder                    # signal → portfolio weights
ModelAPI.available_schedules()               # list schedule configs
```

Six sub-components accessed as `ModelAPI.X`:

| Component | Class | Purpose |
|-----------|-------|---------|
| `Trainer` | `NNTrainer` | Full training loop including CV |
| `Predictor` | `ModelPredictor` | Load checkpoint, run inference |
| `Extractor` | `ModelExtractor` | Extract hidden states / embeddings |
| `Calculator` | `ModelCalculator` | Score as a factor (IC-normalized output) |
| `PortfolioBuilder` | `ModelPortfolioBuilder` | Build portfolio weights from predictions |

---

## `ModelConfig` — configuration hierarchy

Loaded from `configs/model/schedule/{schedule_name}.yaml`.

```yaml
# Example schedule config
model:
  module: nn                     # 'nn' | 'boost' | 'gp'
  arch: GRU                      # architecture name (maps to class)
  hidden_size: 128
  num_layers: 2
  dropout: 0.1

data:
  features: [close, volume, turnover, ...]
  seqlens: [20, 60]              # look-back window lengths
  label: ret_1d                  # prediction target
  universe: hs300

train:
  epochs: 100
  batch_size: 512
  lr: 1e-3
  optimizer: adam
  scheduler: cosine
  early_stopping: true
  patience: 10

swa:
  enabled: true
  strategy: best_k               # best_k | last_k | epoch_k
  k: 5

cv:
  n_splits: 5
  val_years: 1
```

Key fields referenced throughout the codebase:
| Field | Description |
|-------|-------------|
| `model.arch` | Architecture name — matches `NN_ARCHITECTURES` registry |
| `model.module` | Dispatches `AlgoModule` to `nn`, `boost`, or `gp` |
| `data.seqlens` | List of look-back lengths; multiple → multi-scale input |
| `data.label` | Target variable name (from `TRADE` or factor output) |
| `swa.strategy` | SWA weight averaging method |
| `cv.n_splits` | Number of walk-forward CV folds |

---

## `DataModule` — tensor shapes

Wraps data loading for PyTorch Lightning-style training. Input tensors per data type:

| Data Type | Tensor Shape | Notes |
|-----------|-------------|-------|
| OHLCV (daily) | `(batch, seqlen, n_feature)` | seqlen from `data.seqlens` |
| Intraday | `(batch, seqlen, n_inday, n_feature)` | 4D |
| Factor input | `(batch, 1, n_factor)` | single time step |
| Label (target) | `(batch,)` or `(batch, n_horizon)` | regression target |
| Sample weight | `(batch,)` | time-decay or uniform |

When `data.seqlens` has multiple values (e.g., `[20, 60]`), `DataModule` produces a list of tensors — one per seqlen — and the architecture must accept multi-scale inputs.

---

## `BatchInput` (dataclass)

Passed to model `forward()` at every training step:
```python
@dataclass
class BatchInput:
    x: list[torch.Tensor]    # list of tensors, one per seqlen
    y: torch.Tensor          # labels
    w: torch.Tensor          # sample weights
    secid: torch.Tensor      # security IDs (for tracking)
    date: torch.Tensor       # dates (for tracking)
    mask: torch.Tensor       # valid sample mask
```

All model `forward(batch: BatchInput)` methods follow this interface.

---

## 18 NN Architectures

All registered in `src/res/algo/nn/arch/`. Instantiated by name via `ModelConfig.model.arch`.

| Architecture | Key | Type |
|-------------|-----|------|
| GRU | `GRU` | Gated recurrent unit |
| LSTM | `LSTM` | Long short-term memory |
| Transformer | `Transformer` | Self-attention encoder |
| TCN | `TCN` | Temporal convolutional network |
| MLP | `MLP` | Multi-layer perceptron |
| ResNet | `ResNet` | Residual MLP |
| ALSTM | `ALSTM` | Attention-augmented LSTM |
| TRA | `TRA` | Temporal Routing Adaptor |
| HIST | `HIST` | Hierarchical stock transformer |
| TabNet | `TabNet` | Attention-based tabular model |
| DHEN | `DHEN` | Dual-head encoder network |
| SFM | `SFM` | State frequency memory |
| Localformer | `Localformer` | Local attention transformer |
| TimesNet | `TimesNet` | 2D time-series representation |
| PatchTST | `PatchTST` | Patch-based time series transformer |
| iTransformer | `iTransformer` | Inverted transformer for TS |
| Crossformer | `Crossformer` | Cross-dimension transformer |
| DLinear | `DLinear` | Decomposition linear model |

---

## Training Lifecycle

```
ModelConfig.load(schedule_name)
        ↓
DataModule.setup()               # load DataBlock, split CV folds
        ↓
[Walk-forward CV loop — n_splits folds]
  ├── NNTrainer.fit(fold)
  │   ├── epoch loop
  │   │   ├── forward pass (BatchInput → loss)
  │   │   ├── backward + optimizer step
  │   │   ├── callbacks: LR scheduler, early stopping, checkpoint
  │   │   └── SWA accumulation (if swa.enabled)
  │   └── fold checkpoint saved to ModelPath
  └── val metrics logged
        ↓
SWA weight averaging across folds/epochs
        ↓
Final model saved: PATH.model/{schedule}/{fold}/checkpoint.pt
```

---

## Callbacks

All callbacks in `src/res/algo/nn/callbacks/`:

| Callback | Purpose |
|----------|---------|
| `EarlyStopping` | Stop when val loss stops improving (patience=N epochs) |
| `ModelCheckpoint` | Save best checkpoint per fold |
| `LRScheduler` | Cosine / step / plateau LR decay |
| `SWACallback` | Accumulate weights for SWA |
| `MetricLogger` | Write train/val metrics to log file |
| `GradientClip` | Clip gradients to max_norm |

---

## SWA (Stochastic Weight Averaging)

Three strategies, set via `swa.strategy`:

| Strategy | Description |
|----------|-------------|
| `best_k` | Average the K checkpoints with lowest validation loss |
| `last_k` | Average the last K checkpoints (ignoring val loss) |
| `epoch_k` | Average every K-th epoch checkpoint |

`swa.k` controls how many checkpoints are averaged. Higher K = more smoothing = less overfitting, but may underfit on recent data. Typical value: 5.

Final SWA model is saved alongside the per-fold checkpoint.

---

## `ModelPath` — file naming convention

All model artifacts stored under `PATH.model / schedule_name / fold_id /`:

```
PATH.model/
└── gru_ret1d/               # schedule name
    ├── fold_0/
    │   ├── checkpoint.pt    # best single-epoch checkpoint
    │   ├── swa.pt           # SWA-averaged weights
    │   ├── config.yaml      # model config snapshot
    │   └── metrics.json     # training metrics
    ├── fold_1/
    │   └── ...
    └── ensemble.pt          # average across all folds (optional)
```

Predictions are written to `PATH.prediction / schedule_name / {date}.feather`.

---

## `ModelPredictor` / `ModelExtractor` / `ModelCalculator`

```python
pred = ModelAPI.Predictor.predict(
    schedule_name='gru_ret1d',
    dates=[20230101, 20230201],
    universe='hs300'
)
# → pd.DataFrame: index=(secid, date), columns=['pred']

emb = ModelAPI.Extractor.extract(
    schedule_name='gru_ret1d',
    layer='hidden',
    dates=[...], universe='hs300'
)
# → DataBlock: (N_secid, N_date, 1, hidden_size)

factor = ModelAPI.Calculator.as_factor(
    schedule_name='gru_ret1d',
    dates=[...], universe='hs300'
)
# → StockFactor (IC-normalized, same interface as FactorAPI.normalize())
```

---

## Common Patterns / Gotchas

- `data.seqlens` as a list triggers multi-scale input — each element creates a separate tensor in `BatchInput.x`; single-element lists are valid
- SWA is applied **after** training — the `swa.pt` file is what `ModelPredictor` loads by default
- Walk-forward CV folds are non-overlapping in time — `cv.val_years=1` means each fold uses 1 year of validation data
- Predictions are stored per-date — use `DataAPI.build_block()` to reassemble into a DataBlock for portfolio construction
- `AlgoModule` in `src/res/algo/api.py` is the unified dispatcher — it reads `model.module` and imports the correct submodule; don't instantiate NN classes directly
- To add a new architecture: create a class in `src/res/algo/nn/arch/`, inherit from `BaseNNModel`, implement `forward(batch: BatchInput)`; it auto-registers by class name
