# TrainingAnalysisSnapshot

Model-training analysis inputs live at `PATH.interim / 'training_analysis_snapshot'`.
This is a derived data product, not model features, predictions, fitted weights or
portfolio results. All consumers retain the original numerical algorithms.

## Storage and use

`current.json` points to an immutable `generations/<id>/manifest.json`. Each year
contains three LZ4 Feather files, sorted by `(date, secid)`:

- `market.feather`: generator industry, next-session close-to-close/open/vwap
  returns and independent row-presence masks. Missing rows and NaNs remain distinct.
- `evaluation.feather`: default forward return (`nday=10`, `lag=2`, `close`),
  risk industry, default risk-style exposures, label end date and maturity status.
  The existing `DATAVENDOR.nday_fut_ret` calculation is preserved exactly.
- `membership.feather`: original universe membership and available exclusion flags,
  with explicit `universe_member` rows. Benchmark weights retain the original loader.

The manifest lists columns, coverage dates, schema/default contract, source file
metadata and SHA-256 checksums. Feather can be read with `pandas.read_feather`;
industry columns are categorical on disk and restored to object at legacy APIs.
No pickle or per-day files are required. A reader caches at most three annual
frames and uses sorted date offsets for daily slices, without daily full-frame
filtering. A separate DataBlock copy is not persisted.

Each `DetailedAlphaAnalysis` callback keeps an in-memory return-label dictionary
keyed by return parameters, securities, and dates. On the first request it uses
the snapshot when the request matches. Otherwise it calculates labels with
`DATAVENDOR.nday_fut_ret`. Both paths store their results for repeated requests
during the callback, including when the snapshot is absent.

`analysis_snapshot='auto'` opens existing coverage and uses original inputs
for unavailable requests. `off` disables snapshot inputs, and `required`
requires complete, valid coverage. Snapshot preparation belongs to weekly
maintenance or the manual rebuild command. Explicit `t50_backend='legacy'`
and existing generator fallback rules still apply.

## Maintenance

`UpdateAPI.weekly` calls snapshot maintenance **before** model updates on updatable
machines. It covers available daily data from 2017-01-03, preserves existing history,
checks source revisions and rebuilds affected annual partitions. New dates and
newly available future returns trigger rebuilds; unchanged files are hard-linked.
Calendar files and next/label-end mappings
are dependencies too. Source fingerprints use path/size/mtime/ctime; source files
must be updated normally, not altered while preserving all metadata.

Publication and garbage collection share a writer lock. Readers pin generations
with leases under that lock. Publication occurs only after all files are written
and source fingerprints rechecked. Failed builds leave the current pointer intact.
GC retains current, previous and all leased generations; unchanged annual files
share disk blocks. Weekly maintenance logs snapshot failures and continues to
model updates. A missing or unreadable partition in auto mode uses the original
data path for requests that need it.

Schema 3 covers the current calculation semantics. Changes to label algorithms,
risk definitions or stored field meaning require a schema bump and rebuild.
The manifest must never be manually edited to bypass invalidation.

Commands (from the project environment):

```sh
python -m src.res.factor.training_analysis_snapshot inspect
python -m src.res.factor.training_analysis_snapshot validate --start 20240101 --end 20241231
python -m src.res.factor.training_analysis_snapshot rebuild --start 20240101 --end 20241231
```

`rebuild` forces regeneration of the requested plus existing snapshot coverage.
Preparation reports compute, save and total wall time separately. Runtime benchmark
excludes preparation but includes reader opening, source validation and first reads:

```sh
python -m src.res.model.diagnostics.training_snapshot_benchmark --output results/nn_training_review/training_snapshot/year_final
```

The diagnostic uses frozen real `gru_day_V1` predictions and compares all factor
result tables plus the portfolio account with tolerances `rtol=1e-6, atol=1e-8`.
Its previous path uses the stage-2 monthly market package. It measures analysis
replay, not neural-network fitting, plotting, export or full train_schedule time.

## Historical data limitation

Local historical universe files lack complete `warnst` flags. The snapshot does
not invent them, and default universe requests fall back when required flags or
dates are missing. Local benchmarks use the same stored four-flag universe on both
sides; they do not establish equivalence for unavailable historical `warnst` data.
Membership imports existing universe exports; their original generation/quality
remains the responsibility of the upstream universe pipeline.

Historical local benchmark projected files disagree with raw constituent reconstruction
on some dates. Benchmark weights are deliberately excluded from the snapshot until
that upstream discrepancy is resolved; this change preserves existing analysis results.

## Manual preparation

Run from the repository root:

```sh
.venv/bin/python scripts/data/create_training_analysis_snapshot.py
.venv/bin/python scripts/data/create_training_analysis_snapshot.py --resume false
.venv/bin/python scripts/data/create_training_analysis_snapshot.py --start 20240101 --end 20241231
```

The default `resume=true` preserves existing coverage and rebuilds changed years.
`resume=false` recomputes the requested available interval without reusing old
partitions. Publication remains atomic, so a failed rebuild preserves the current
snapshot. Every successful build, including an unchanged update, cleans old and
interrupted generations. Retention uses the published previous-generation pointer,
not directory access times. Current, previous and actively leased versions survive.
A later update removes versions whose readers have finished. Concurrent training
falls back to original inputs while the writer lock is held.

## Stage 3 acceptance — 2026-11-04

The default callback configuration uses `analysis_snapshot=auto`. Missing,
stale, incomplete or unreadable inputs use the original data path. Return labels
are memoized within each callback for both snapshot hits and computed misses.
Weekly preparation failures are logged and model updates continue.

Acceptance covers default/nondefault labels, absent snapshots, missing partitions,
source revisions, incremental coverage, failed publication, reader leases, writer
contention, repeated-update retention and interrupted-generation cleanup.
The manual script was exercised against real 2024 data with both resume settings.
The full rebuild produced identical hashes for all three files. The following
unchanged update performed no computation or writes and retained two generations.

Five paired fresh-process replays of 242 trading days in 2024 produced identical
factor result tables and portfolio accounts (rtol 1e-6, atol 1e-8).
Both paths used the latest callback-scoped label memoization.
Median seconds for the previous monthly package and annual snapshot respectively:

- Factor evaluation: 23.566 and 14.887.
- FMP inputs: 4.700 and 3.365.
- Portfolio generation excluding inputs: 0.178 and 0.117.
- Accounting excluding inputs: 0.529 and 0.428.
- Total FMP: 5.595 and 4.036.
- Core analysis replay: 34.524 and 20.525.

These measurements exclude snapshot preparation, prediction loading, plotting,
export and neural fitting. Historical local warnst coverage is incomplete; both
paths used the same stored four-flag universe. Benchmark weights use the original
loader because historical projected weights disagree with raw reconstruction.

The production portfolio entrypoints share generator_array.py. The former
t50_array.py module and its obsolete documentation have been removed. Historical
contract tests and monthly package diagnostics remain as compatibility/reference
coverage. Raw benchmark artifacts and snapshot data remain local, outside Git.

After consolidating the generator module, all 42 related regression tests passed.
An additional full-2024 paired replay passed all 11 output-table comparisons.
Its core replay times were 41.895s and 20.407s, with FMP times 7.329s and 4.452s.
This final pair verifies integration; the five-pair medians above remain the
performance summary because single-run timing is variable.
