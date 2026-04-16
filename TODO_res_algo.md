# TODO — src/res/algo

Bugs and improvement opportunities identified during the documentation pass (2026-04-16).
No code has been changed — these are observation notes only.

---

## Bugs

### 1. `FactorVAE.py` — `VAESampling.reparameterize()` addition instead of multiply

**File:** `src/res/algo/nn/model/FactorVAE.py`

The reparameterization trick is `z = mu + sigma * eps`, but the current
implementation uses `mu + sigma + eps` (addition).  This makes the sampled
latent `z` depend linearly on noise magnitude rather than being scaled by
the learned standard deviation — the model cannot learn a meaningful posterior
variance.

**Fix:** change `mu + sigma + eps` → `mu + sigma * eps`.

---

### 2. `multiloss.py` — `GLS.multi_losses()` wrong method name, never overrides base class

**File:** `src/res/algo/nn/loss/multiloss.py`

`GLS` (Gradient Loss Scaling) defines `multi_losses()` but the base class
method that the training loop calls is named `multihead_losses()`.  The
override is never dispatched: GLS silently falls back to equal weighting
at runtime.

**Fix:** rename `GLS.multi_losses` → `GLS.multihead_losses`.

---

### 3. `MLP.py` — list `hidden_size` path has no inter-layer activations

**File:** `src/res/algo/nn/layer/MLP.py`

When `hidden_size` is provided as a list of integers, the layers are stacked
without any activation function between them, making the whole stack
effectively a single linear transformation regardless of depth.

**Fix:** insert an activation call (e.g. `get_activation_fn(activation)`)
between each pair of linear layers in the list path.

---

## Code Quality

### 4. `ABCM.py` — filename/class name mismatch

**File:** `src/res/algo/nn/model/ABCM.py`

The file is named `ABCM.py` but the class inside is `Astgnn` (Adaptive
Spatial-Temporal Graph Neural Network).  The registered key in `AVAILABLE_NNS`
is `'abcm'`.  This creates confusion when navigating the codebase.

**Recommendation:** either rename the class to `ABCM` or rename the file to
`Astgnn.py` and update `AVAILABLE_NNS`.

---

### 5. `mod_gru` duplication — defined in both `RNN.py` and `ABCM.py`

**Files:** `src/res/algo/nn/model/RNN.py`, `src/res/algo/nn/model/ABCM.py`

`ABCM.py` defines its own local `mod_gru` function that duplicates the one
already in `RNN.py`.  Any divergence between the two will be silent.

**Fix:** remove the local copy in `ABCM.py` and import from `RNN.py`.

---

### 6. `TFT.py` — local `MultiHeadAttention` duplicates `layer/Attention.py`

**File:** `src/res/algo/nn/model/TFT.py`

TFT defines its own `MultiHeadAttention` class rather than reusing
`layer/Attention.MultiheadAttention`.  Any bug fixes or improvements to the
shared layer will not propagate to TFT.

**Recommendation:** refactor TFT to import from `layer/Attention.py` or at
minimum add a comment noting the duplication.

---

### 7. `EwLinear` naming — computes a mean, not a linear transform

**File:** `src/res/algo/nn/layer/basic.py`

`EwLinear` applies element-wise weights and then averages across the time
dimension — it is a temporal mean pool, not a linear transformation.

**Recommendation:** rename to `EwMean` or `TemporalMeanPool` to avoid
confusion with `nn.Linear`.

---

### 8. `BoostInput.from_dataframe()` — undocumented last-column-is-label assumption

**File:** `src/res/algo/boost/util/boost_io.py`

`from_dataframe` hard-codes the label as `iloc[:, -1:]` with no parameter to
override it.  Any DataFrame where the label is not the last column will
silently produce wrong results.

**Fix:** add an explicit `label_col` parameter (defaulting to `None` which
falls back to the current last-column behaviour).

---

## Performance / Architecture

### 9. `FactorVAE.py` — `FactorPredictor.forward()` Python loop over `factor_num`

**File:** `src/res/algo/nn/model/FactorVAE.py`

`FactorPredictor.forward()` iterates over `factor_num` with a Python loop.
On GPU this creates `factor_num` sequential kernel launches instead of one
batched operation.

**Fix:** vectorise using `torch.stack` + a single batched matrix multiply or
grouped convolution.

---

### 10. Patch utilities duplicated across `PatchTST`, `TSMixer`, `ModernTCN`

**Files:**
- `src/res/algo/nn/model/PatchTST.py`
- `src/res/algo/nn/model/TSMixer.py`
- `src/res/algo/nn/model/ModernTCN.py`

`create_patch()` and `random_masking()` are copy-pasted in all three files.
Any bug fix must be applied in three places.

**Recommendation:** extract both functions into `src/res/algo/nn/layer/basic.py`
or a new `layer/patch.py` and import from there.

---

### 11. `AlgoModule.export_available_modules()` — file write at every import

**File:** `src/res/algo/api.py` (bottom of file, module level)

`AlgoModule.export_available_modules()` is called unconditionally at import
time.  This writes a file to `PATH.temp` on every `import src.res.algo`,
including during unit tests, CI, and any script that merely introspects
the module.

**Fix:** guard with an environment variable or make it an explicit opt-in call
(e.g. remove the bottom-of-file call and let callers invoke it when needed).

---

### 12. `SSAMF.update_mask()` — uses `CrossEntropyLoss` (wrong for regression)

**File:** `src/res/algo/nn/optimizer/sam.py`

`SSAMF` (Sparse SAM with Fisher mask) calls `CrossEntropyLoss` to compute
Fisher importance scores.  For regression tasks (which is the primary use
case here) this is semantically wrong — the Fisher information should be
computed under the actual task loss (MSE or IC-based).

**Fix:** accept a `criterion` parameter in `SSAMF` and default to the task
loss rather than hard-coding `CrossEntropyLoss`.

---

### 13. `VariableSelectionNetwork` — no assertion on `input_dim % num_vars == 0`

**File:** `src/res/algo/nn/model/TFT.py`

`VariableSelectionNetwork` assumes `input_dim` is evenly divisible by
`num_vars` (each variable gets `input_dim // num_vars` features).  When this
does not hold the last variable silently receives fewer features due to
integer division truncation.

**Fix:** add `assert input_dim % num_vars == 0, f"input_dim {input_dim} must be divisible by num_vars {num_vars}"`.
