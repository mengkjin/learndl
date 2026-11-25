# Unified training scheduler for the next version

The release after 0.7.11 provides an opt-in trial of shared training
admission. Production rollout depends on the trial results.

## Admission policy

Use four training classes in descending priority.

1. **Production continuation.** The 00:10 Asia/Shanghai trigger submits a
   batch when registered models are due and training data is ready.
2. **Manual training.** Include standalone model and schedule scripts. A
   project fit with no explicit automatic classification defaults to manual.
3. **Lifecycle upgrades.** Explicit one-shot declarations build and validate
   isolated candidates. Production consumers use the exact identity in Git.
4. **Idle worklist.** Preserve source ordering, revision handling and the
   existing idle-resource tests.

Admission runs before preparing the model directory for fit. It uses
request identity, priority, readiness, required resources and queue order.
The nightly batch keeps admission across production model transitions.
During the trial, measure wait causes and queue time and check for starvation.

At a completed `model_num × model_date` boundary, a checkpoint-aware trainer
checks for eligible higher-priority work, verifies saved weights and publishes
a continuation before yielding. Resume with the original priority, directory
and verified configuration. Revalidate lifecycle intent and worklist source
revisions. Explicit stops and superseded requests stay stopped. Fits outside
the checkpoint protocol can block tracked resources until they exit.

Keep kernel fit and resource locks, per-model locks and lifecycle validation.
Preserve code and maintenance leases for patches, Git updates and authorized
installations. Inference and standalone tests stay outside the fit queue.

## Trial and acceptance

- Use observation mode on the CUDA server to compare the unified policy's
  decisions with existing dispatch. Switch to unified mode to test all four
  priorities, then return to legacy mode.
- Test admission races between each adjacent priority pair, queued requests
  with unavailable data and research attempting to enter between two
  production models.
- Verify checkpoint identity after pause, recovery after a launcher crash or
  reboot, configuration changes while paused, explicit stops, deduplicated
  notifications and preservation of the approved prediction directory.
  Turning off the trial must preserve history and let active work finish.

## Switch modes on the training host

After installing Watchdog with this source version, open **Non-Research
Operations → Training Scheduler Mode** in the CLI Hub. The same controls are
available from the checkout.

```bash
.venv/bin/python -m src.automation.scheduling.training_scheduler status
.venv/bin/python -m src.automation.scheduling.training_scheduler mode observe
.venv/bin/python -m src.automation.scheduling.training_scheduler mode unified
.venv/bin/python -m src.automation.scheduling.training_scheduler mode legacy
```

New installations start in `legacy`. The runtime status shows the active and
requested modes, the requests blocking a transition, queued work, and resumable
checkpoints. A mode change takes effect when the admitted jobs release their
leases. New fit requests wait during that handoff. The mode selection and
checkpoint records survive restart. The `legacy` command restores the original
policy after the same handoff. Paused promotions can finish in their original
candidate directories after returning to `legacy`.

Use `observe` to see recent admission and checkpoint decisions alongside the
existing dispatch. Use `unified` to trial all four priorities. Foreground manual
training displays its request ID and wait reason. Ctrl+C during this wait
cancels the request. Manual fits that do not contend for the exclusive NN
resource can still overlap. Equal-priority requests contending for the same
resource use their original submission order, including after a pause.

For CUDA acceptance, start a worklist fit and make an eligible lifecycle
upgrade available. Verify a checkpoint handoff, then submit a manual fit and
verify the upgrade resumes in the same candidate directory. Let a due
production batch contend with the manual fit. Verify that research stays
outside the batch until the last model finishes. Repeat with unavailable data,
a changed declaration, and an explicit stop. Switch back to `legacy` during a
trial run and verify that the old policy takes over once active work releases
admission. Use the production registration workflow in
[model lifecycle](../strategy/MODEL_LIFECYCLE.md#one-shot-upgrades) if you later
decide to deploy an accepted candidate.
