# Callback measurement

The accepted callback changes are integrated; there are no callback fastpath
switches. Old saved `train.experimental_callback_fastpath` settings are ignored.
The independent first-batch `train.experimental_metrics_fastpath` setting remains
unchanged.

- `AttemptMetrics.latest()` reads the last epoch table without traversing history.
- `BasicTestResult` groups once and calls `basic_group_stats()` in its module.
  The helper expects rows with nonmissing predictions and both evaluation labels;
  it preserves average ties, sample standard deviation and original bucket edges.
- Hidden RankIC uses the ordinary metric component and shared Spearman ranking:
  each input with sorting length >= 5000 uses scatter. Custom calculators remain
  intact. This shared ranking optimization predates the callback integration;
  it is not an additional callback speed gain. No wrapper copies or separate
  threshold setting are needed.
- Retraining and callback dispatch retain their original paths. Early stopping
  retains the separately approved conditional short-circuit calculation; it has
  no experiment switch or cache.

## Server execution

```sh
uv run python scripts/5_test/benchmark_nn_callbacks.py --device cuda
uv run python scripts/5_test/benchmark_nn_callbacks.py --device cuda --short-tests
```

The microbenchmark compares the three integrated paths with independent reference
calculations: full-history reads, five groupby passes, and double-argsort ranking.
Reference calculations live only in the diagnostics module, not in training.
Use `--only summary_latest basic_test` to focus the run. Hidden cases default to
2000/4095/4096/4097/5000 rows and 32/64 dimensions; the production threshold stays
5000. The `--sizes` option changes test sizes, not production behavior.

Five warmups precede at least nine alternating timed blocks. Isolated CUDA timing
synchronizes at block boundaries. JSON preserves raw samples, comparison errors,
call counts, absolute savings, variability, device and Git metadata; CSV summarizes
the cases and the console log records failures. A microbenchmark pass is evidence
for that case only, not proof of end-to-end training improvement.

`--short-tests` runs four identical default-model short tests with a fixed seed
and the first-batch metric switch disabled. These are repeatability/profile runs,
not old/new A/B runs. It compares decisions, weights, diagnostics and exported
results and reports stage times without asserting before/after gains. Each run
records actual callbacks, effective configuration, diagnostic rounds, detailed
analysis, export submission/completion and host waits. It preserves source and
model evidence. Hook timings are host timings without per-hook GPU synchronization;
async completion latency includes queueing and I/O. Fresh runs do not measure
snapshot recovery. Existing historical A/B evidence remains in the results folder.

The reusable `CallbackProfile` context manager lives in
`src/res/model/diagnostics/callback_profile.py`; it patches only during measurement
and restores methods on exit. Normal training does not enable this profiler.

## Mail

Mail is sent through `src.proj.util.web.task_mail` after logs close, with JSON,
CSV, log and zipped evidence attached. Use `--no-email` for local verification.
A delivery failure preserves all results and can be retried without training:

```sh
uv run python -m src.proj.util.web.task_mail <output>/mail
```

Output defaults to a new folder under `results/nn_training_review`; an existing
`--output` directory is rejected to protect previous evidence. No scheduled job
is installed.
