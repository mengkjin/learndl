# Universe prepared data — stage 1

`PreparedUniverse(root, name).prepare(dates)` imports finalized daily snapshots
into monthly Feather partitions. It preserves previously prepared dates, reads
only new/changed daily snapshots, rewrites only affected months, and hard-links
unchanged months into an immutable generation. A no-change request publishes
nothing. A single-writer lock and atomic pointer protect publication.

`load(dates, exclusions)` pins one generation, checks daily source fingerprints
and upstream dependencies before and after reading, and verifies every requested
exclusion column. Missing columns are not converted to False. Schema v2 requires
a new root when migrating from the experimental v1 files.

Dependencies include listing/rename information, same-day and closest valuation
files, loser and warning exposures, and each date's preceding 90 calendar days
of announcements across three exchanges. Missing paths are fingerprinted too,
so late-arriving files invalidate old data. Information changes conservatively
invalidate all affected requests; announcement changes invalidate their rolling
windows. Size, mtime and ctime identify ordinary local filesystem revisions;
this is not a cryptographic audit of a storage system that preserves all metadata.
Only all/top universes are supported by the prepared store in this phase.

On an upstream revision, `prepare` refuses to re-certify stale daily snapshots.
Use `prepare_from_sources(dates, verified_warning_history=True)` after verifying
announcement-source coverage. It resets relevant vendor/exclusion caches,
rebuilds only uncovered/revised daily universes, and then publishes partitions.
Without verified coverage it fails before source writes. Initial `prepare`
imports trust finalized daily snapshots; dependency capture cannot retroactively
prove their historical provenance. Absent historical warnst columns remain an
explicit coverage failure. Normal sources/load failures propagate without
publishing partial generations. Superseded generations are retained for rollback.

`Universe(name, prepared_root=root, prepared_mode='required').get(dates)` uses
this store. `auto` falls back on absent/incomplete prepared files, but upstream
staleness fails explicitly rather than falling back to stale process caches.
`off` remains the default in this phase. Explicit preparation handles incremental
repair; consumers do not silently authorize missing-source risk calculations.

DetailedAlphaAnalysis accepts `prepared_universe_root` and
`prepared_universe_mode`. Its preload constructs the Portfolio once and passes
that object explicitly to top/screen/revscreen/reinforce. Reuse lasts only that
callback invocation. Direct FMP calls construct once for their exact factor
dates, not every date left in a process-global cache. The model training snapshot uses annual membership partitions by default.
This monthly format remains available for explicit legacy configurations.
