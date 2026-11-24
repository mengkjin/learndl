# Model lifecycle

This manager owns NN/boost training lifecycles; non-fitting linear/factor models
retain their existing lifecycle implementations.

The policy and public Python interface live in `src/res/model/util/lifecycle.py`.
The CUDA server's daily update calls this module to resume due production models.

## Exploration

New NN schedules use `min(original max_epoch, 20)`,
`min(original window, 1200)` and every second original training node. Boost
schedules retain their fitting parameters and select every second node. The
original interval and calendar anchor are not changed. Global2Top callbacks run
after these limits and may increase max_epoch. Short tests retain their existing
short-test settings and cannot be promoted.

The model's `configs/production.yaml` contains the uncompressed, resolved model
and algorithm configuration. `configs/lifecycle.yaml` contains lifecycle state
and the per-model `budget` mapping. Existing models keep their historical budget
unless explicitly enabled. Editing a live schedule does not update this target.

```python
from src.res.model.util.core import ModelPath
from src.res.model.util.lifecycle import ModelLifecycle

manager = ModelLifecycle(ModelPath('gru@research_model').base)
manager.configure_budget(max_epoch=15, window=800, step=3)
status = manager.inspect()
# Explicit replacement of the FULL model/algo/boost_head target snapshot:
# manager.update_target(updated_snapshot)
```

Change budgets between runs. `enabled=False` disables all exploration compression;
`step=1` disables only node thinning. Setting NN limits at least as high as the
original values permits full single-node training with sparse dates.

Frozen factor-sequence folds keep their original training nodes and purged split
boundaries. Quick fitting intersects their train/validation dates with the reduced
available window and fails explicitly if either partition becomes empty.

## Selecting a mode through schedules and worklists

The schedule script exposes `training_mode` (`auto`, `quick`, `sparse`, `full`),
`training_step`, `quick_max_epoch` and `quick_window` in its interactive parameter
form. The same settings are accepted by `ModelAPI.schedule_model` as
`training_mode=...` and `lifecycle_options={...}`.

`configs/schedule/worklist.yaml` accepts mixed strings and mappings:

```yaml
resume: true
force: false
fit:
  - existing_schedule
  - name: reusable_experiment
    mode: sparse
    resume: true
    lifecycle:
      step: 2
  - name: fast_experiment
    mode: quick
    resume: false
    lifecycle:
      max_epoch: 15
      window: 800
      step: 3
```

Mappings require `name`; optional fields are `resume`, `force`, `short_test`,
`start`, `end`, `mode` and `lifecycle`. The lifecycle mapping accepts `mode`,
`step`, `max_epoch`, `window`. Unknown fields, conflicting mode aliases and
conflicting duplicate schedule entries are rejected. Identical repeats are
collapsed, preserving list order.

Entry fields override global worklist defaults. Lifecycle fields merge by key.
Within the selected model, explicit settings override its saved budget; omitted
settings retain that budget. New models retain the defaults described above.
`auto` keeps the saved mode. `sparse` ignores epoch/window caps and preserves full
single-node fitting. `full` disables node thinning as well. Neither mode changes
the lifecycle stage or grants production approval. Short tests retain smoke
settings; validation/production models reject exploration overrides.

Switching quick to sparse with `resume: true` does not certify quick weights as
full weights: the manifest compares actual fitting semantics and retrains
incompatible nodes. When the budget changes, old prediction/report outputs are
archived to `snapshot/lifecycle_previous_evaluation.*.tar` before evaluation caches
are reset; weight files remain available for compatibility checking. The existing worklist revision policy is preserved (including comment changes).
Worklist settings remain in revision checks, so changed
options invalidate completion and stale queued configurations are not executed.
Abnormal-exit recovery may force resume, as before, to retain completed work.

## Production registration and daily updates

The `prediction` section of `configs/strategy/model.yaml` selects production
directories by exact identity, including an index only when the name specifies
one. Several prediction aliases may use different members of the same directory.
Each directory gets one update check. Hidden extraction and exploration schedules
serve their own purposes and do not register production models. The optional
`lifecycle` mapping can supply `start` and `end` bounds for a listed identity.

On the CUDA server, daily_update prepares prediction data, checks registered
models and resumes due training before it computes predictions and portfolios.
The weekly job extends historical fit dumps. The idle research worker leaves
production training to daily_update.

Model dates follow the saved calendar anchor and interval. A rolling fit uses
samples through the trading day at offset `-1 - skip_horizon` from its model date.
The update uses `both` data, which can extend the historical dump through the
latest available market data. The trainer checks that the cutoff has finite
labels and that the label horizon fits the historical window.

## Resume and acceptance

Training resumes in the exact prediction directory. Its manifest allows
compatible weights to be reused and its prediction recorder resumes saved
evaluation. Legacy weights with matching saved settings are recorded when the
first update runs. Old indexed candidates remain on disk as historical runs.

Acceptance checks every requested date, model number and submodel. For dates with
observable out-of-sample trading days it also checks prediction coverage and finite
values. The update writes `snapshot/evaluation_manifest.json` and releases the
model for production use after acceptance. An interrupted in-place update stays
pending for the next daily check. Per-node training evidence is stored in
`snapshot/model_manifest.json` and transitions in `logs/lifecycle.jsonl`.

`manager.update_target(full_snapshot, retrain=True)` explicitly replaces the
resolved target and invalidates existing weights. The next update archives the
previous saved schedule before applying the replacement.

Deployment follows the existing project release workflow.
