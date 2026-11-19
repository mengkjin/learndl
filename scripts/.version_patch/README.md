# Version patches for local project files

Use a version patch when a release changes the format or layout of local files such as
`models/` and `data/`. Commit the patch and the new `src/__version__.py` together.
The existing Autofix queue is for individual operational repair requests.

On each server, publish the updated scheduler and bootstrap with
`bash runs/install_schedules.sh plan --host <host> --diff` and the corresponding
`apply --host <host>` in an idle maintenance window. The Git maintenance timer
applies version patches after checkout verification or detects manual pulls on
its next tick. It checks for version patches even if Git fetching is disabled.
Automatic research, Autofix, and daily, weekly and RCQuant updates share the
same code lock. Each worker takes a read lock and checks the patch ledger before
starting a script, including when invoked by an older launcher. A patch needs
exclusive access. Missing, pending or failed migrations keep workers waiting
between attempts. Calendar timers can still register their slots. After
verification the next tick uses the existing queue, overlap and expiry rules.
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
  same locks as the timer. On a fresh installation, the first maintenance tick
  or `apply` registers the current code version and skips older patches. Empty
  directory scaffolding and the checkout's `results/README.md` do not count as
  existing data.
- Restore the original version ledger and its backups alongside imported data.
  Before installing or starting tasks on a host with imported files and no
  ledger, run `python -m src.automation.version_patch init --baseline <data-version>`
  with the actual version of the files. The system blocks until this is set.
  The check includes data, models, results, local Optuna and TensorBoard files,
  and task mail artifacts. File timestamps are not used to infer the version.
- After a failure, inspect the attempt under
  `PATH.runtime/version_patch/attempts/` and check the patch's backup.
  Correct the cause and run
  `python -m src.automation.version_patch retry 0.7.2/001` using the reported
  patch ID. The command resumes the first unfinished patch and then later
  patches. An interrupted attempt requires the same explicit retry.
- Retry can use corrected committed code on a clean fast-forward descendant
  of an already verified checkout, as long as the failed patch ID and the
  successful patches are preserved. Task admission resumes after the next
  Git maintenance verification if a checkout was still pending.
- The mail worker retries delivery independently. The email identifies the
  failed attempt and log path. Restore a missing or damaged ledger from backup
  before resuming. Keep the ledger with the host data when recovering a server.
  A code downgrade below the data version requires an operator-managed restore.

Hosts upgraded from the previous name keep using their existing
`PATH.runtime/version_autofix/` directory. The new CLI reads the same history
and backups in place. Use the new CLI and catalog path for new work.
