"""One-shot NN metric A/B requests handed from autofix to the idle GPU worker."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def requests_dir():
    from src.proj import PATH
    return PATH.lc_machine / 'nn_metrics_ab' / 'requests'


def submit(request_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', request_id):
        raise ValueError('Invalid request ID')
    path = requests_dir() / f'{request_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {'id': request_id, 'schema_version': 1}
    try:
        with path.open('x') as stream:
            json.dump(document, stream)
    except FileExistsError:
        if json.loads(path.read_text()) != document:
            raise ValueError('Existing request differs')
    return path


def observe(store):
    from .idle_worklist import OWNER, version
    for path in sorted(requests_dir().glob('*.json')):
        doc = json.loads(path.read_text())
        if doc != {'id': path.stem, 'schema_version': 1}:
            raise ValueError(f'Invalid NN A/B request: {path}')
        key = version(doc)
        if any(r.get('task_kind') == 'nn_metrics_ab' and r['version'] == key
               and r['phase'] != 'deferred' for r in store.owned(OWNER)):
            continue
        return {'task': f'nn_metrics_ab:{doc["id"]}', 'task_kind': 'nn_metrics_ab',
                'version': key, 'revisions': doc, 'request_id': doc['id']}
    return None


def execute(request, store):
    from src.proj import PATH
    from src.proj.util.script import FitLock
    from src.proj.util.cli.script_session import as_script_main
    from src.proj.util.filesys.dynamic_import import dynamic_modules
    from .recovery import record
    if FitLock.is_held():
        return 75
    # Keep the shared slot occupied across all four subprocesses. Each trainer
    # owns its normal exclusive NN lock; holding that lock here would deadlock it.
    guard = FitLock.guard()
    guard.__enter__()
    script = PATH.scpt / '4_train' / '5_nn_metrics_ab.py'
    previous = sys.argv
    try:
        sys.argv = [str(script)]
        with as_script_main(script):
            module = next(iter(dynamic_modules(script)))
            task = module.main(request_id=request['request_id'], email=True)
        success = bool(task.success and task.execution_success)
        record(store, request['id'], script_outcome='success' if success else 'error')
        return 0 if success else 1
    finally:
        sys.argv = previous
        guard.__exit__(None, None, None)
