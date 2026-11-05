"""One-shot factor-sequence validation requests handed from autofix to the idle GPU worker."""
from __future__ import annotations

__all__ = [
    'requests_dir',
    'submit',
    'observe',
    'execute',
]

import json
import re
import sys


def requests_dir():
    from src.proj import PATH
    return PATH.lc_machine / 'factor_sequence' / 'requests'


def submit(request_id, resume_from=None, cleanup_legacy=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', request_id):
        raise ValueError('Invalid request ID')
    if resume_from is not None and (not re.fullmatch(r'[A-Za-z0-9_-]+',resume_from) or resume_from==request_id):
        raise ValueError('Invalid resume source')
    path = requests_dir() / f'{request_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {'id': request_id, 'schema_version': 1}
    if cleanup_legacy:
        if resume_from:
            raise ValueError('Cleanup requires a new suite, not legacy resume')
        document.update(schema_version=3,cleanup_legacy=True)
    if resume_from:
        document.update(schema_version=2,resume_from=resume_from)
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
        valid_v1=doc=={'id':path.stem,'schema_version':1}
        valid_v2=(set(doc)=={'id','schema_version','resume_from'} and doc['id']==path.stem
                  and doc['schema_version']==2 and isinstance(doc['resume_from'],str)
                  and re.fullmatch(r'[A-Za-z0-9_-]+',doc['resume_from']) and doc['resume_from']!=path.stem)
        valid_v3=doc=={'id':path.stem,'schema_version':3,'cleanup_legacy':True}
        if not (valid_v1 or valid_v2 or valid_v3):
            raise ValueError(f'Invalid factor-sequence request: {path}')
        key = version(doc)
        if any(r.get('task_kind') == 'factor_sequence' and r['version'] == key
               and r['phase'] != 'deferred' for r in store.owned(OWNER)):
            continue
        return {'task': f'factor_sequence:{doc["id"]}', 'task_kind': 'factor_sequence',
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
    # Keep the shared slot occupied across all validation subprocesses. Each trainer
    # owns its normal exclusive NN lock; holding that lock here would deadlock it.
    guard = FitLock.guard()
    guard.__enter__()
    script = PATH.scpt / '4_train' / '6_factor_sequence_verify.py'
    previous = sys.argv
    try:
        sys.argv = [str(script)]
        with as_script_main(script):
            module = next(iter(dynamic_modules(script)))
            try:
                resume_from=request.get('revisions',{}).get('resume_from')
                if request.get('revisions',{}).get('cleanup_legacy'):
                    from .factor_sequence_restart import execute as restart
                    restart(request['request_id'],module)
                elif resume_from:
                    from .factor_sequence_recovery import recover
                    if recover(request['request_id'],resume_from,module)==75:
                        return 75
                else:
                    module.main(run_id=request['request_id'])
            finally:
                # Record only confirmed delivery; otherwise supervisor retains its alert.
                receipt = PATH.result / 'factor_sequence' / request['request_id'] / 'mail_status.json'
                if receipt.exists() and json.loads(receipt.read_text()).get('status') == 'sent':
                    from .runtime import record_script_email
                    suite = json.loads((receipt.parent / 'suite.json').read_text())
                    record_script_email(success=suite['status'] == 'passed' and not suite.get('training_mail_failed',0))
        record(store, request['id'], script_outcome='success')
        return 0

    finally:
        sys.argv = previous
        guard.__exit__(None, None, None)
