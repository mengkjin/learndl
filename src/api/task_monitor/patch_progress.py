"""Read patch progress from the ledger and publish it while task sampling is paused."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time

import portalocker

from src.automation import version_patch
from src.automation.console import logger
from src.shared.file_ops import atomic_json

__all__ = ['FILENAME', 'snapshot', 'publish', 'publisher']

FILENAME = 'version-patch.json'


def _execution_lease_held(runtime):
    # A surviving child can own the lease after its supervisor exits. A free
    # read lock proves the ledger's running row is awaiting reconciliation.
    try:
        with (runtime / 'scheduling' / 'git_update.lock').open('rb') as stream:
            portalocker.lock(stream, portalocker.LOCK_SH | portalocker.LOCK_NB)
    except portalocker.exceptions.LockException:
        return True
    except FileNotFoundError:
        pass
    return False


def snapshot(root: Path, runtime: Path) -> dict[str, str | int | float | None] | None:
    try:
        ledger = version_patch.ledger_path(runtime)
        if not ledger.parent.exists():
            return None
        state = version_patch.status(root, runtime)
        attempt = state['failed']
        recorded_only = attempt and attempt['status'] == 'running' and not _execution_lease_held(runtime)
    except (version_patch.MigrationError, OSError, sqlite3.Error, ValueError, SyntaxError) as exc:
        return dict(status='unavailable', error=str(exc), sampled_at=time.time())

    pending = state['pending']
    if attempt:
        phase = 'unconfirmed' if recorded_only else attempt['status']
    elif state['ready']:
        phase = 'ready'
    elif state['baseline'] is None:
        phase = 'uninitialized'
    elif any(version_patch.version_key(row['patch'].split('/')[0]) > version_patch.version_key(state['applied'])
             for row in state['records']):
        phase = 'partial'
    else:
        phase = 'pending'
    result: dict[str, str | int | float | None] = dict(
        status=phase, applied=state['applied'] if state['baseline'] else None, target=state['target'],
        remaining=len(pending), sampled_at=time.time())
    if pending:
        result.update(patch=pending[0].id, script=Path(pending[0].relative_path).name)
    if attempt:
        result.update(attempt=attempt['attempt'], started=attempt['started'], finished=attempt['finished'],
                      error=attempt['error'])
        if phase == 'failed':
            try:
                saved = json.loads(Path(attempt['log_path']).with_name('failure.json').read_text())
                if isinstance(saved, dict) and isinstance(saved.get('error'), str):
                    result['error'] = saved['error']
            except (OSError, ValueError):
                pass  # Older attempts only have the ledger error.
    return result


def publish(root, runtime, config):
    """A missing share or failed status read must never change a patch outcome."""
    from .remote_transport import task_key

    try:
        path = config.shared_root / 'runtime' / task_key(config.source_host) / FILENAME
        progress = snapshot(root, runtime)
        if progress is not None:
            atomic_json(path, dict(host=config.source_host, **progress))
        elif path.exists():
            path.unlink()
    except Exception as exc:
        logger.error('Could not publish version patch status: %s', exc)


def publisher(root, runtime):
    from .remote import settings

    try:
        config, enabled = settings(runtime)
    except Exception as exc:
        logger.error('Could not load remote monitor settings: %s', exc)
        return None
    if enabled:
        return lambda: publish(root, runtime, config)
    return None
