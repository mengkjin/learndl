# Daily Update Workflow

The current entry is `scripts/1_autorun/0_daily_update.py`, calling `UpdateAPI.daily()`.
Do not use the retired `scripts/1_data` / `scripts/2_factor` examples.

## Data prerequisites

`DataAPI.update()` runs core downloads, sellside downloads, risk-model updates,
registered custom updaters, then announcements. Within DailyQuote, validated
`trade_ts/day` is published before the shared adjusted-price service runs. This
intentional completion hook does not import or invoke CustomUpdater classes.

The later AdjPriceUpdater checks source/output receipts and normally skips.
Only the first trading date on/after the fixed `START_DATE=19970101` can omit a
previous price panel. Missing predecessor prices trigger backward discovery and
chronological construction. Missing raw trading-day history is an explicit error;
a run's `start` parameter cannot establish a new seed date.

Queries retain automatic repair through the same price service and reload the
persisted result. Source publication and chained writes share a process lock;
outputs use atomic replacement. The registered ListingPriceUpdater refreshes the
full listing-reference table after adjusted-price prerequisites. Updater dependency
ordering is declared explicitly, with cycles and missing dependencies rejected.

## Downstream execution

`UpdateAPI.daily()` stops downstream model work when `DataAPI.is_updated()` fails.
It then orchestrates factor/model/trading and reporting updates through their APIs.
Standalone factor updates use `scripts/3_factor/0_update_factors.py`; data block
preparation uses `scripts/2_data/0_train_data.py`. Rollback uses
`scripts/1_autorun/2_rollback_update.py` and retains the project's recomputation
semantics rather than restoring a point-in-time database snapshot.

## Maintenance and release

Explicit data repairs live under `src/data/maintenance`, with thin entry scripts in
`scripts/8_fixes`. Do not run production rebuilds as an import or deployment step.
Quiesce in-flight jobs before updating source and installed module commands together.
Keep runtime ledgers, locks, request IDs and historical manifests in place; rollback
requires the matching old source and installation snapshot, not mixed entrypoints.
