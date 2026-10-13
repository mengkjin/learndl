# Linear baseline schedules

Each YAML in this directory defines one fixed linear combination. Normal results
are stored in `models/linear/<schedule_name>/`; short tests use the existing
`models/st/linear@<schedule_name>/` isolation. A model directory alone does not
register a linear model: the defining schedule must still exist.

Ready-to-run schedules (choose a date range covered by your local sources):

- `gru_only`: GRU prediction alone, normal/equal.
- `gru_df_equal`: GRU + `df_scores_v0`, normal/equal.
- `gru_df_weighted`: the same sources, normal/[0.7, 0.3].
- `gru_df_raw_sum`: the same sources, none/[1, 1] for raw addition.
- `pred_mix`: the same combination as `gru_df_equal`, retained as a starter template.

These schedules are not added to the worklist automatically. Copy one to a new
schedule name for each experiment. For percentile preprocessing, set
`final.precursor: percentile` under `input`.

An archived-model input can also be selected:

```yaml
model:
  module: linear
input:
  type: final
  final.types:
    - pred@my_prediction
    - model_pred@gru@my_experiment@1@0@best@pred.0
  final.precursor: normal
  final.weights: equal
  final.refresh: false
```

The script menu entry `5_test/3_test_linear.py` lists only linear schedules,
supports `start`/`end`, and defaults to `resume=False`. It writes normal results
under `models/linear/<schedule_name>/` and includes a schedule config editor.

Or run through the existing trainer:

```python
from src.res.model.model_module.application import ModelTrainer

ModelTrainer.schedule('gru_df_equal', short_test=False, resume=0,
                      start=20240102, end=20241231)

trainer = ModelTrainer.initialize(schedule_name='gru_df_equal', short_test=False)
trainer.go(resume=1)
```

For an isolated short trial, explicitly override the project's default short-test
dates as well as the label-loading date filter:

```python
trainer = ModelTrainer.initialize(
    schedule_name='gru_df_equal', short_test=True,
    override={'model.beg_date': 20250324, 'model.end_date': 20250328,
              'input.filter.date': '20250324~20250328',
              'train.callbacks': ['BasicTestResult'], 'env.mem_storage': True})
trainer.go(resume=0)
```

`linear` is **not** a null model. Its canonical fullname is `linear@pred_mix`,
its internal module identity is `linear@linear`, and `requires_fit=False` controls
its `data + test` lifecycle independently of null-model path/config behavior.
It has one output, model number 0 and submodel `best`, and needs no checkpoint.
There is no algorithm weight sweep or fitting stage in this version.

## Input sources

- `pred@name`: one stored prediction column from `DB_pred`; never reruns a model.
  Published predictions may already average multiple model numbers.
  If DataBlock reshaping fails, duplicate rows are checked and their dates are
  warned about. Identical rows are collapsed in memory and conversion retries;
  conflicting values for the same stock/date raise `ValueError` with the source,
  requested period and example keys. Source files are not modified, and normal
  conversions do not incur this duplicate check.
- `factor@name`: one registered factor, loaded without extra normalization.
- `model_pred@module@name@directory_index@model_num@submodel[@pred.N]`:
  one archived model output. Directory 1 is the unnumbered experiment directory.
  Model references share hidden's parser, including its abbreviated defaults
  (directory 1, number 0, checkpoint `best`). Select `pred.N` for multi-output models.

List different model numbers/checkpoints as separate inputs. For each date,
`model_pred` uses the selected number's latest archive strictly before that date;
it never falls back to a future archive. Dates before the first checkpoint are
missing. Existing raw prediction caches are reused and missing dates computed;
`final.refresh: true` refreshes requested dates once per data-module run.
Only nn/boost upstream models are supported; recursive linear sources are not.

## Preprocessing and weights

Every source contributes one scalar column. Stocks and dates are aligned before
processing. Each day, wholly empty components are ignored. The remaining
nonzero-weight components define the common finite stock universe; individual
missing stock values are not filled. A day without any available component is
skipped. The effective test period is bounded by usable dates (including saved
dates when resuming). The final layer does not forward-fill values.
Factor loaders keep their own existing semantics: `df_scores_v0`, for example,
uses its primary/alternative sellside databases and carries the last usable
cross-section onto later dates. Coverage is reported. Zero-weight sources do not restrict that universe.

`final.precursor` applies independently to each column on the complete daily
cross-section, before weighting and batching:

- `normal` (default): population Z-score followed by clipping to [-3.5, 3.5],
  without a second standardization. Constant finite cross-sections become zero.
- `percentile`: average tied rank / finite count, in (0, 1].
- `none`: raw values.

`final.weights: equal` means the mean of that day's available components (1/N
per available column). If only one component is available, its value is used.
A numeric vector such as
`[1, 1]` or `[1, -1]` is used literally, without automatic normalization.
It is an input option, so a plain vector is not expanded into multiple models.
For raw prediction addition, choose `none` and `[1, 1]`.

The final loader also works with nn/boost and combo inputs, applying preprocessing
and weights to the individual feature columns. Linear itself accepts only final
inputs. Final sequence lengths and steps must be 1.

## Explicit resume

The default is a fresh evaluation, including when the model directory exists.
`is_resumable` stays false so automatic checkpoint discovery does not enroll
linear models in daily resumes. Only an explicit `resume=1` enables continuation;
there is no schedule resumability switch and no interactive resume question.

Resume requires saved configuration. Input/weight/precursor or evaluation-setting
changes are rejected to avoid mixing experiments; extend the end date or use
`resume=0` to recompute. A fresh run replaces this schedule's generated test
records/results, retaining upstream models and raw prediction caches.

Final inputs are read and preprocessed once in the data stage. For linear
resume, saved prediction records determine the exact unfinished dates first,
including historical gaps; completed dates are not read again. No new dates
means no final-source reads. Dates confirmed to have no usable combination are
recorded too, so leading/trailing empty periods do not reload on every resume.
Use `resume=0` to reconsider those dates after upstream data is backfilled.
The test stage only consumes prepared tensors,
even though results are still organized into monthly model dates.

Schedule names must be unique across current, shared and linear directories.

## Missing sources and shared computation

Nonzero-weight source storage and model checkpoints are checked before a fresh
run clears previous results. Missing storage raises `FileNotFoundError`; unknown
factors or invalid configuration raise `ValueError`, with the source and context.
Empty daily components are ignored and dates with no usable combination are
skipped and summarized. An entirely empty component is allowed when another
component has data. If the whole evaluation has no usable combination (and no
saved predictions on resume), the data stage raises an error. Storage missing
from the computer is still an error, rather than being treated as an empty day.
Empty individual months are allowed. Resume with no new dates is a no-op.

Final inputs and `Amodel.combine_linear` share `src.func.linear` weight parsing
and per-component multiplication. Their defaults remain intentionally explicit:
linear uses the finite intersection and literal vector coefficients;
AlphaComposite keeps its component preprocessing, date fallback, union/zero
alignment, normalized weights and final zscore. Equal settings on aligned values
produce equal contributions; existing strategy defaults are not changed.
Normalized weights with a zero total raise `ValueError`; literal linear weights
such as `[1, -1]` remain supported.
