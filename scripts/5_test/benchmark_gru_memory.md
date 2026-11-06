# Synthetic GRU CUDA memory benchmark

Run from the repository root on the CUDA server:

```bash
.venv/bin/python scripts/5_test/benchmark_gru_memory.py
```

The default run measures 22 cases: 5,000 stocks × 30 days × 16 bars × 6 features,
intraday hidden sizes 32 and 64, two GRU layers, dropout 0.1, FP32, and three
complete training steps per case. The day GRU always has hidden size 64.
Weights and data are random. The head mirrors the default `gru_gru` linear /
BatchNorm / linear / BatchNorm structure, using synthetic MSE and Adam.
The script does not load production data or models, change worklists, or run
the training scheduler. Run on an otherwise idle GPU for comparable results.

Each case uses a fresh subprocess; an OOM is recorded and the next case starts.
Cases have a 1,200-second timeout, configurable with `--timeout`. Default modes:

- `full`: all stock/day sequences in one GRU call.
- `full_checkpoint`: one checkpoint around that entire intraday encoder.
- `chunk`: ordinary chunks of 50,000, 25,000 or 12,500 stock/day sequences.
- `chunk_outer_checkpoint`: one checkpoint around the entire chunk loop.
- `chunk_checkpoint`: a separate checkpoint around each chunk.

All modes preserve stock/day ordering and the full-stock day GRU, BatchNorm and
loss. Checkpoint uses `use_reentrant=False, preserve_rng_state=True`; it is active
immediately in the selected cases (not waiting for an OOM callback).
The day GRU is uncheckpointed in all cases by default. `--day-checkpoint` enables
it equally across all modes, allowing a second comparison closer to production
recovery. Dropout assignments may differ between chunk sizes, so outputs are not
expected to be bitwise equal across these performance cases.

Encoder results use a compact clone of the last time step in all default modes.
This avoids an output view retaining full sequence storage and does not detach
gradients. `--retain-output-view` runs the corresponding original-view experiment;
use this with `full` / `full_checkpoint` when examining the existing encoder.

## Larger workload and resident data simulation

```bash
.venv/bin/python scripts/5_test/benchmark_gru_memory.py \
  --stocks 5000 6000 --hidden 64 --resident-gib 8 --day-checkpoint
```

`--resident-gib 8` allocates and touches an additional 8 GiB GPU buffer, held
throughout each case. This simulates a fixed baseline, not production allocator
fragmentation or data pipeline behavior. It is included in reported peaks. Set
it using observed resident allocations, not the total training peak (which would
double-count activations). Avoid concurrent training: external GPU use affects
available capacity but is not included in this process's allocated-memory peak.

Useful options: `--steps 5`, `--recipient ADDRESS`, `--device cuda:1`, `--no-email`,
`--output-dir /absolute/new/directory`. The output directory must not already exist.
To keep running after disconnecting, use a terminal multiplexer or:

```bash
nohup .venv/bin/python -u scripts/5_test/benchmark_gru_memory.py > /tmp/gru_memory_benchmark.log 2>&1 &
```

## Results and email

Results default to `logs/gru_memory/<UTC timestamp>/`:

- `summary.txt`: readable results, including OOM/error stage.
- `results.csv`: comparison rows.
- `results.json`: configuration, environment, per-stage allocations and timings.
- `case_*.json`, `case_*.result.json`, `case_*.log`: isolated worker inputs/results/logs.
- `mail/`: existing project `TaskResultMail` delivery receipts and prepared payloads.

After the experiment, the script sends the summary and three report attachments
using the server's existing email credentials. The default recipient is the
configured sender's own address. No credentials are embedded in this script.
Partial results are also mailed after handled errors or interruption. SIGKILL,
power loss, or host failure cannot guarantee a final email; completed cases are
saved incrementally. Disabled email or an unconfirmed send is explicitly reported
and returns a nonzero exit status. For an SMTP failure with a prepared receipt,
retry only delivery, without rerunning the benchmark:

```bash
.venv/bin/python -m src.proj.util.web.task_mail /absolute/results/directory/mail
```

The benchmark exits zero after a completed experiment even if some/all cases OOM;
OOM is an expected measurement outcome. Worker errors, timeouts and email failures
return nonzero. A successful mail means SMTP acceptance, not verified inbox delivery.

## Interpreting measurements

Peak allocated/reserved memory includes setup and all steps, including the cold
first step and lazy Adam state initialization. Each step resets peak statistics
after clearing gradients; stage peaks are cumulative within that step. Allocated
memory at forward completion helps reveal retained autograd state, but includes
the fixed baseline and all other live tensors. It is not a direct measurement of
cuDNN reserve space. Warm step time is the median excluding step 0 and includes
explicit GPU synchronization at each phase.

An OOM peak is only the allocation achieved before failure: the failed request is
not included. Do not label a smaller OOM peak as a successful memory reduction.
Compare successful cases at identical dimensions, baseline and day checkpoint
settings. Lower reserved memory is not necessarily lower live memory. JSON also
records device free/total memory, PyTorch/CUDA/cuDNN versions and allocator config.

This experiment measures synthetic capacity and the memory/time tradeoff. It does
not prove that production training fits, nor find an exact maximum batch size.
Use `--stocks` with several values to bracket the capacity threshold, then validate
the chosen implementation against the real failing workload.
