# Round 3: repaired data, three ordinary idle schedules

Local worklist order preserves `resnet_gru_simp_15m`, then adds:

1. `factorseq_v3_snapshot`
2. `factorseq_v3_temporal`
3. `factorseq_v3_contextual`

Each is an ordinary sparse schedule with a new model identity. Existing v2 schedules,
weights and datasets remain intact. Training still uses the project model-level
recording, report and mail lifecycle; no custom training launcher is added.

## Dataset selection and shared freeze

All three schedules use the campaign key `repaired_20261117`. On its first schedule
resolution, select the completed repair run with the latest `affected_end` (then the
most recently written completion summary), from `results/factor_sequence_fixes`.
The run must have `status: completed` and `reports/dataset.json`. Failed, unfinished
and dry-run entries do not qualify. Verify the repaired bundle's checksums and exact
frozen factor/feature order; a corrupt selected bundle is an error, not a reason to
silently select older data. The old `20261031_factor_sequence_verify_01` suite is not
a fallback.

Freeze dataset path, manifest SHA256, data version, mature-label SHA256 and all folds
in `.local_resources/shared/factor_sequence/repaired_20261117.json` on that host.
Subsequent modes reuse the pin even if another repair finishes. Changes to the frozen
manifest or labels cause an error. A later experiment with a different dataset must
use a new campaign key and new schedule identities rather than deleting a pin and
mixing inputs within the same experiment.

Selection happens at runtime because the repaired dataset is produced on the server.
The repository configuration alone does not assert that server repair has completed.
No separate historical audit job or model training was launched during implementation.

## Common experiment settings

- Target `std_lag1_10`, seed 17, MSE, equal weights and single-date sequential batches.
- OOS starts at the first trading day on/after 20170101 and ends at the latest mature
  label supported by the frozen repaired data. Missing daily labels inside the
  selected train/validation/test dates fail rather than silently shortening history.
- Train expands to 1260 days; validation 252 days before exact 11-session endpoint
  purging; 239-day initial input warm-up; non-overlapping 63-day test windows.
- Maximum 30 epochs, validation peak patience 5, best checkpoint, hidden dimension
  32, existing dropout/GRU settings and residual/FiLM structure unchanged.
- DetailedAlphaAnalysis tasks `[factor, t50]` and FactorResidualDiagnostics remain on.

`compact` selects 132 stock channels: value/valid/gap/never for 33 factors. Only
Contextual loads the shared 132-channel factorperf (66 values + validity) and
30-channel factormarket (15 values + validity). Snapshot uses one stock date;
Temporal/Contextual use 60, shared context 120. No new complete-case sample filter
or blanket forward fill is introduced.

The repaired NPY loader selects date, stock and feature axes together before
materializing the requested panel; it does not first copy all 332 stock channels.
This reduces temporary copies but does not establish server host/GPU memory acceptance.

## Verification and publication boundary

Tests cover ordinary ScheduleConfig resolution, all three model input widths and
forward calls, immutable shared pinning, missing/failed repair rejection, changed
label rejection, expanding windows, purge boundaries and worklist ordering.
All changes remain local; no Git commit/push or server workload launch is performed.
