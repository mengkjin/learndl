# Callback efficiency experiment

All five switches default to false; the first-batch metrics fastpath is independent.
Current baseline already includes Spearman scatter dispatch at length 5000. Do not
count that merged change as a new callback gain.

## Server execution

From the repository root:

```bash
uv run python scripts/5_test/benchmark_nn_callbacks.py --device cuda --short-tests
```

Nine alternating microbenchmark blocks precede four independent default short tests
in old/new/new/old order. Both arms explicitly disable the metrics fastpath. Each
candidate must have lower median time and at least 7/9 faster blocks in every applicable
case before entering the new arm. No eligible candidate means four baseline controls.
The child verifies effective switches after configuration flattening.

For microbenchmarks only:

```bash
uv run python scripts/5_test/benchmark_nn_callbacks.py --device cuda --only summary_latest basic_test
```

`--sizes` controls diagnostic input lengths. Defaults cover 2000, 4095, 4096, 4097,
5000, hidden widths 32/64, and thresholds 4096/5000. `--threshold 4096` changes which
threshold governs eligibility; default is 5000. `--repeats` must be at least 9.
Use `--no-email` locally; `--output` must be a new directory. Explicit CUDA requests
fail when unavailable. No scheduler or automatic server run is created.

## Configuration

`train.experimental_callback_fastpath` contains:

- `summary_latest`: last-table reads for six total-metric summaries.
- `summary_hidden_rank`: diagnostic-only ranking at `hidden_rank_threshold`.
- `decision_local`: unused convergence branch elimination and one-decision best reuse.
- `basic_test`: single grouped pass, shared average-tie ranking and standardization.
- `dispatch`: ordered hook lists, rebuilt at initialization and model restart.
- `hidden_rank_threshold`: positive integer, default 5000; five switches default false.

Nested and flattened dotted keys are supported. Missing keys remain off. Custom
hidden calculators retain their existing path. No changes to training policies,
setup, HiddenCorr loss, diagnostic frequency, aggregation, or TensorBoard copying.
DetailedAlphaAnalysis, StatusDisplay, CallbackTimer and LearnRateReset are measured,
not rewritten. Disabled callbacks are not credited as savings.

## Evidence and email

Outputs live under `results/nn_training_review/callbacks_<timestamp>/`:
`report.json`, `summary.csv`, `console.log`; with short tests, per-child logs,
effective configs, profiles, model artifacts and source snapshots are also retained.
The CSV covers microbenchmarks. Raw block samples, call counts, errors, selected
flags and full-training observations are in JSON. Child `wall_seconds` excludes
imports; parent `process_seconds` includes them.

Closed logs, CSV, JSON and a zipped evidence bundle are mailed through the shared
`src.proj.util.web.task_mail` implementation, including failed runs. Model outputs
are included, market input caches are not. Retry failed delivery without training:

```bash
uv run python -m src.proj.util.web.task_mail <output-directory>/mail
```

## Timing and acceptance

Profiles report host inclusive/exclusive durations, not isolated GPU kernel time.
Do not sum nested inclusive durations. Only isolated GPU microbenchmark blocks
synchronize; full runs do not synchronize every hook. Temporary measurement patches
are restored on success/failure and are never installed during normal training.

Profiles distinguish ordinary/diagnostic train+validation intervals, hidden rank,
corrcoef, aggregation, metric reads, TensorBoard operations/copying, factor/portfolio
analysis, plots, export submission and async waits. Async completion latency includes
queueing and I/O after submission, not worker compute alone. Existing detailed-analysis
logs retain loading substeps. Four default short tests are fresh executions, not
proof of resumed-analysis performance.

Events, model selection, weights, predictions and configuration (apart from run names
and experimental switches) must agree. Hidden diagnostics allow rtol=1e-6/atol=1e-7;
basic statistics allow rtol=1e-10/atol=1e-12, with matching NaNs and table structure.
Tolerances are never automatically widened. Every run, including outliers, is kept.
Overlapping or slower full-run timings are labelled “未证实提速”; two runs per arm
cannot establish statistical significance. Production defaults remain off.
