# Unified training scheduler for the next version

Design the shared admission policy and run a limited trial in the release
after 0.7.11. Production rollout depends on the trial results.

## Proposed policy

Use four training classes in descending priority.

1. **Production continuation.** The 00:10 Asia/Shanghai trigger submits a
   batch when registered models are due and training data is ready.
2. **Manual training.** Include standalone model and schedule scripts. A
   project fit with no explicit automatic classification defaults to manual.
3. **Lifecycle upgrades.** Explicit one-shot declarations build and validate
   isolated candidates. Production consumers use the exact identity in Git.
4. **Idle worklist.** Preserve source ordering, revision handling and the
   existing idle-resource tests.

Use one durable admission decision immediately before fit. Base it on
request identity, priority, readiness, required resources and queue order.
Preserve the nightly batch reservation across production model transitions.
Set a fairness policy during design and measure wait causes and queue time
during the trial.

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

- Design request states, lease ownership and recovery rules before enabling
  dispatch. Reuse the existing persistent run records and checkpoint tickets
  where they meet those rules.
- Start in observation mode on the CUDA server. Compare proposed admission
  decisions with existing dispatch. Enable enforcement for selected trial
  models through an installer setting that can restore the existing policy.
- Test admission races between each adjacent priority pair, queued requests
  with unavailable data and research attempting to enter between two
  production models.
- Verify checkpoint identity after pause, recovery after a launcher crash or
  reboot, configuration changes while paused, explicit stops, deduplicated
  notifications and preservation of the approved prediction directory.
  Turning off the trial must preserve history and let active work finish.
