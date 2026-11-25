# Version patches for local project files

Use a version patch when a release changes the format or layout of local files such as
`models/` and `data/`. Commit the patch and the new `src/__version__.py` together.
The existing Autofix queue is for individual operational repair requests.

Every release starting with v0.7.11 requires an entry in the root
[CHANGELOG.md](../../CHANGELOG.md). Record the effects of each migration and
any deployment or verification steps operators need.

On each server, publish the updated scheduler and bootstrap with
`bash runs/install_schedules.sh plan --host <host> --diff` and the corresponding
`apply --host <host>` in an idle maintenance window. The Git maintenance timer
applies version patches after checkout verification or detects manual pulls on
its next tick. It checks for version patches even if Git fetching is disabled.
Automatic research, Autofix, and daily, weekly and RCQuant updates share the
same code lock. Each worker takes a read lock and checks the patch ledger before
starting a script, including when invoked by an older launcher. A patch needs
exclusive access. With a registered baseline and an intact checkout, a pending
version upgrade allows daily updates and other scripts to start using the current
data. Maintenance retries when those scripts release their leases. A manual
script reports the pending version range on stderr. If it must wait for
maintenance, it reports the reason immediately and again when the reason changes.
An active patch, an incomplete patch version, a failed attempt or an unreadable
ledger still holds new work. Calendar timers can register their slots during
maintenance. Workers resume with the existing queue, overlap and expiry rules.
Watchdog can finalize earlier runs between patch attempts.
Use an idle window for manual Git operations and stop external processes that
write the same files before updating.

## Patch contract

Use `<version>/<number>_<name>.py`, for example `0.7.2/001_rename_config.py`.
Versions are numeric `major.minor.patch`. Numbers have at least three digits and
are unique per version. The operator ID for the example is `0.7.2/001`.
Keep successfully released patches at their original paths with their original
bytes. Use a higher version for repairs to migrations that have already succeeded.
The worker reads committed Python files and checks them against the checkout.

Each script defines `apply(ctx)` and `verify(ctx)`. The context has
`root`, `source_version` (last fully applied patch version), `target_version`
(the patch's version), `log_path`, and `backup_dir`. `verify` must return
`True` to record success. The patch gets a new backup directory and log for every
attempt. Include the usual script metadata header. The migration runner owns
locking, logging, and failure mail for these callbacks. Use the project logger
for diagnostics. Avoid logging secrets.
Keep code used only for one release beside its patches. Use
`<version>/support/<name>.py` for helpers shared with a preflight command or
migration test. The runner checks these helpers against the committed checkout
without treating them as patch entrypoints. Run subprocesses synchronously in
the patch process group. Detached processes are outside the migration contract.

Use atomic replacement for small config files. For directory moves on the same
filesystem, validate both paths and use rename instead of copying a large
dataset. Decide explicitly how already migrated files, missing files, and
conflicts should behave. Keep enough backup information to recover after a
partial run.

A version directory may contain `timeouts.json`, for example
`{"001_rename_config.py": 120}`. Values are positive integer seconds. The default
is 1,800 seconds per patch.

Illustrative patch for a single JSON model config. The actual scope and key names
should be chosen for the release.

```python
# coding: utf-8
# author: Release maintainer
# date: 2026-11-19
# description: Rename a model configuration key
# content: Back up the local model config and migrate old_key to new_key.
# email: False
# mode: os

import json
import os
import stat
import tempfile


def apply(ctx):
    path = ctx.root / 'models' / 'example' / 'config.json'
    if not path.exists():
        return  # This host has no copy of this model.
    original = path.read_bytes()
    config = json.loads(original)
    if 'old_key' in config and 'new_key' in config:
        raise ValueError(f'Conflicting keys in {path}')
    if 'old_key' not in config:
        return  # Already migrated, or verification will report a missing new_key.
    (ctx.backup_dir / 'config.json').write_bytes(original)
    config['new_key'] = config.pop('old_key')
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', delete=False) as stream:
        temporary = stream.name
        try:
            os.fchmod(stream.fileno(), stat.S_IMODE(path.stat().st_mode))
            json.dump(config, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def verify(ctx):
    path = ctx.root / 'models' / 'example' / 'config.json'
    if not path.exists():
        return True
    config = json.loads(path.read_bytes())
    return 'new_key' in config and 'old_key' not in config
```

## Operation and recovery

Run from the project checkout with the project's Python environment.

- `python -m src.automation.version_patch status` shows versions, pending
  patches, and attempt history. `plan` shows pending work.
- `python -m src.automation.version_patch apply` runs pending patches under the
  same locks as the timer. `apply` and `retry` report the current patch, log path,
  and verification result as the run progresses. Progress goes to stderr so
  stdout remains a single JSON result. Each fully verified batch with at least
  one patch records a single summary email for watchdog delivery, listing the
  applied patches and resulting data version. On a fresh installation, the first maintenance tick
  or `apply` registers the current code version and skips older patches. Empty
  directory scaffolding and the checkout's `results/README.md` do not count as
  existing data.
- Restore the original version ledger and its backups alongside imported data.
  Before installing or starting tasks on a host with imported files and no
  ledger, run `python -m src.automation.version_patch init --baseline <data-version>`
  with the actual version of the files. The system blocks until this is set.
  The check includes data, models, results, local Optuna and TensorBoard files,
  and task mail artifacts. File timestamps are not used to infer the version.
- A failed attempt reports the underlying exception, affected paths when
  available, and the next retry command. `status` exposes failure details and
  the notice's delivery state. The watchdog retries SMTP independently. Inspect
  `patch.log`, `failure.json`, and the `backup/` directory alongside it. For
  `0.7.2/001`, the first blocked move in `backup/experiment-results.json` names
  the source and destination. `lsof` diagnostics identify open files and their
  processes, and separately report missing tools, timeouts, and probe errors.
  Desktop portal or GVFS mount warnings outside the selected tree do not
  block the move when the probe otherwise reports no holders.
- After fixing the cause, run
  `python -m src.automation.version_patch retry 0.7.2/001` using the reported
  patch ID. The same command resumes an interrupted attempt.
- Retry can use corrected committed code on a clean fast-forward descendant
  of an already verified checkout, as long as the failed patch ID and the
  successful patches are preserved. Task admission resumes after the next
  Git maintenance verification if a checkout was still pending.
- With automatic Git updates enabled, a recorded patch failure still allows
  a clean fast-forward. The worker automatically retries the first unfinished
  patch once on each newly verified descendant commit. Set
  `watchdog.jobs.git_auto_update.retry_failed_patch_on_update: false` to require
  explicit retries. To retry the same commit after an operational fix, run the
  reported `version_patch retry` command.
- Restore a missing or damaged ledger from backup before resuming. Keep the
  ledger with the host data when recovering a server.
  A code downgrade below the data version requires an operator-managed restore.

### Interface maintenance

Manual runs and the automatic maintenance worker share the code and maintenance
locks. Keep the watchdog available for failure notices and CLI recovery.

Before applying patches, the runner records the running managed console,
managed panels and dashboard instances, and any registered desktop CLI
reporting an idle menu. It stops those processes by recorded identity. Existing
CLI sessions from older code may need a manual exit because their idle state
is unavailable. A service or external process that still holds a file can
appear in the patch diagnostic.

After a failure, the original console and Task Center are restored for
diagnostics. After success, the remaining recorded interfaces are restored
using their original ports and migrated sources. The desktop CLI returns to
its menu through its registered desktop recovery session. If a port is busy,
a source cannot be resolved, or the desktop session is unavailable, the
migration remains applied and the interface recovery report lists the problem.
The automatic Git maintenance worker retries residual `ready` interface recovery
under the maintenance and code leases, even when no data patches are pending.
Before a subsequent migration, incomplete recovery defers execution without
creating a failed patch attempt. Once recovery succeeds, pending patches proceed
normally. Actual patch failures still require the existing retry workflow.
Use `python -m src.automation.version_patch restore-services` for an immediate
manual recovery attempt after fixing the reported condition.

### Migration checks for upgrades through 0.7.9

- **0.7.2/001** verifies TensorBoard moves against file hashes, then moves or
  copies experiments. Optuna uses SQLite backup and retains the original
  database and its sidecars. Identical copies can survive a retry. Conflicting
  contents require inspection before retry.
- **0.7.2/002** fixes the original seven-day mail window and reuses durable
  artifact delivery identifiers. Confirm the publisher configuration and
  inspect `recent-mail.json` for missing or unverifiable packages.
- **0.7.3/001** moves factor residual intermediates to model snapshots.
  Already moved source directories disappear from the next discovery.
  Two existing copies are preserved for inspection. Both released entry
  point fingerprints are accepted after the open-file probe update.
- **0.7.4/001** uses a persistent move journal and per-tree fingerprints to
  resume after renames. Check destination conflicts and preserve the journal
  when investigating a partially applied layout migration.
- **0.7.5/001** renames the synchronized model archive and retargets saved
  references. A populated old and new archive requires inspecting both trees.
- **0.7.6/001** builds model properties and verifies published or durably
  prepared historical metadata. It keeps a source inventory and delivery
  journal for retry. Ambiguous historical identities are reported.
- **0.7.7/001** backs up saved YAML before renaming the status callback.
  The runner accepts successful attempts for the two known released
  fingerprints of this exact entry point.
- **0.7.7/002** consolidates legacy model batches into build indexes, remaps
  references, and keeps original batches in the attempt's backup directory.
  Inspect content verification and publication errors before resuming.

- **0.7.9/001** removes securities outside every known listing interval from
  daily all/top universe snapshots. Only changed files are backed up and replaced.
  The patch keeps surviving rows in order and does not refill rankings.

### Data repairs in 0.7.11

- **0.7.11/001** replays the configured tracking and backtest portfolios that
  use `df_scores_v0`, `df_scores_v2`, or `df_scores_v3` from their first saved
  rebalance on or after 2026-01-01. This covers the Dongfang cutoff and uses the
  corrected Huayuan fallback and portfolio freshness rules. The earlier holdings
  seed the replay and stay in place. The platform server rebuilds the holdings,
  accounts, analysis reports and existing daily CSV entries for the affected
  tracking portfolios. Original files and output checksums are in the attempt
  backup. Receiver hosts get verified replacements through the production
  artifact channel.
- **0.7.11/002** rebuilds the 18 stock-factor categories from stored factor
  data. Fit dumps resume after their last usable date, starting no later than
  the first trading day after 2026-04-10. Predict dumps are rebuilt for the
  current prediction window. Each written dump is checked for usable values
  on its latest stored source date. The attempt backup includes the original
  dumps and `factor-datablocks.json` with the verified dates.

If `0.7.11/002` stops at `liquidity/fit` with a string arithmetic error,
preserve the first attempt's backup. From the affected checkout, this read-only
inspection prints the first example of each extra column, text signal or
unparseable signal per factor in the failed window:

```bash
.venv/bin/python - <<'PY'
import pandas as pd
from src.res.factor.calculator import FactorCalculator

for calc in FactorCalculator.iter(category1='liquidity'):
    seen = set()
    name = calc.factor_name
    for date in calc.stored_dates(20260413, 20261124):
        data = calc.Load(date)
        for col in data.columns.difference(['date', 'secid']):
            series = data[col].dropna()
            if series.empty or (col == name and pd.api.types.is_numeric_dtype(series)):
                continue
            issue = 'extra column'
            if col == name:
                invalid = series[pd.to_numeric(series, errors='coerce').isna()]
                issue = 'numeric text' if invalid.empty else 'invalid value'
                if not invalid.empty:
                    series = invalid
            signature = (col, issue, str(series.dtype))
            if signature not in seen:
                print(name, date, *signature, repr(series.iloc[0]))
                seen.add(signature)
PY
```

After the corrected commit is verified on the server, check
`python -m src.automation.version_patch status`. If the installed Git worker
does not retry automatically and the patch is still failed, use
`python -m src.automation.version_patch retry 0.7.11/002`. Confirm verification
and business admission in `status`. The successful attempt writes
`factor-datablocks.json` in its own backup directory.

Hosts upgraded from the previous name keep using their existing
`PATH.runtime/version_autofix/` directory. The new CLI reads the same history
and backups in place. Use the new CLI and catalog path for new work.
