# Model lifecycle

This manager owns NN/boost training lifecycles; non-fitting linear/factor models
retain their existing lifecycle implementations.

The policy and public Python interface live in `src/res/model/util/lifecycle.py`.
Training, prediction and automation call this module; queue ownership and retry
admission remain in the existing automation scheduler.

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

## Promotion intent

Add exact model identities to `configs/strategy/model.yaml`:

```yaml
lifecycle:
  gru@research_model:
    stage: validation
  lgbm@tree_model:
    stage: production
    start: 20170103
    end: 99991231
```

`validation` requests full fitting and evaluation but does not publish a production
alias. Changing it to `production` authorizes automatic publication after successful
acceptance. No second confirmation is required. Existing prediction and hidden
extraction registrations are treated as legacy production intent unless overridden
by an explicit lifecycle entry. Do not register short-test identities.

Daily reconciliation and the automatic training worker inspect these registrations.
The worker must be enabled on the training host. Production repairs have priority
over validation and ordinary exploration; tasks use the existing research lane,
maintenance admission, bounded retry policy and model lock. Repeated checks do not
mutate the worklist YAML or enqueue duplicate requests.

`ModelLifecycle.inspect_registered()` returns pending repair plans and records legacy
audits. It is a reconciliation operation, not a purely read-only status query.
`manager.inspect()` reads one model's current state, target and artifact manifest.

## Repair and serving

Repairs create an existing-style `@index` model instance and keep the approved
instance unchanged. Compatible independent nodes are copied; missing or incompatible
nodes are trained. Transfer-dependent and hidden/final-dependent repairs conservatively
retrain their chain. Ordinary managed resumes fill individual holes and propagate
transfer invalidation to subsequent nodes.

The candidate always performs a fresh full evaluation over the selected interval.
A scope beginning between training nodes includes its preceding node to cover the
requested evaluation period. Production acceptance requires complete weights and
finite predictions covering every required date, model number and submodel, and
successful completion of the existing test/report pipeline. No return or IC threshold
is imposed. Choosing production in the strategy file expresses the research decision.

Publication atomically changes the approved reference after evaluation succeeds.
The original model's lifecycle file retains the previous approved reference and the
candidate directory for diagnostics. Canonical model names accompany absolute paths
so the approved alias can resolve under another host's data root. Old directories
are retained; there is no automatic artifact deletion. Failed repairs reuse their
candidate directory on retry, retain completed compatible weights and reset candidate
evaluation outputs.

`stage`, `status`, `approved_model`, `validated_model`, `accepted_request` and
`candidate_path` in the lifecycle file show progress. Per-node provenance is in
`snapshot/model_manifest.json`; acceptance evidence is in the candidate's
`snapshot/evaluation_manifest.json`; transitions are appended to `logs/lifecycle.jsonl`.

Legacy production continues serving while evidence is incomplete. Matching saved
legacy weights can be reused, with their provenance explicitly marked legacy.
Newly trained or changed weights are never published before candidate acceptance.
Daily integrity checks use artifact size/mtime and manifest integrity; fitting/reuse
checks hash weight content. Upstream model files are tracked by size/mtime. Historical
raw-data revisions are not independently versioned by this module; after a data
correction, call `manager.update_target(full_snapshot, retrain=True)` to require fresh weights
and revalidation. An explicit target update also creates a new request generation,
even if its parameter values are unchanged.

Deployment uses the project's existing scheduler installation/publication workflow.
A source checkout alone does not publish installed automation code.
