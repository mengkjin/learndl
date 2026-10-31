"""Explicit one-shot RL resume requests, executed by the shared idle worker."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import portalocker


def request_dir(root):
    from src.proj import MACHINE
    return Path(root)/'.local_resources'/MACHINE.name/'rl_experiment_resume/requests'


def validate_request(doc, root):
    if not isinstance(doc, dict):
        raise ValueError('Resume request must be a mapping')
    if doc.get('schema_version') != 1 or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', doc.get('request_id', '')):
        raise ValueError('Invalid resume request ID/schema')
    run = Path(doc['run_directory']).resolve()
    if not run.is_relative_to(Path(root).resolve()/'results/rl_experimental'):
        raise ValueError('Resume target must be inside project RL results')
    for field in ('config_sha256', 'identity_sha256'):
        if not re.fullmatch(r'[a-f0-9]{64}', doc.get(field, '')):
            raise ValueError(f'Invalid {field}')
    if not re.fullmatch(r'[a-f0-9]{40,64}', doc.get('source_commit', '')):
        raise ValueError('Invalid original source commit')
    if type(doc.get('no_email')) is not bool or (doc.get('recipient') is not None and not isinstance(doc['recipient'], str)):
        raise ValueError('Invalid resume mail settings')
    return doc


def submit(root, request_id, plan):
    """Called by autofix after audit; no RunStore writes or GPU work here."""
    doc = dict(schema_version=1, request_id=request_id, run_directory=plan['run'],
               source_commit=plan['source_commit'], config_sha256=plan['config_sha256'],
               identity_sha256=plan['identity_sha256'], recipient=plan.get('recipient'), no_email=False)
    validate_request(doc, root)
    directory = request_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{request_id}.json'
    with portalocker.Lock(directory/'.requests.lock', timeout=0):
        if path.exists():
            if json.loads(path.read_text()) != doc:
                raise ValueError('Resume request is immutable; use a new request ID')
        else:
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
            temporary.replace(path)
    return path


def candidate(store, root):
    for path in sorted(request_dir(root).glob('*.json')):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            doc = validate_request(json.loads(raw), root)
            if path.stem != doc['request_id']:
                raise ValueError('Request ID differs from filename')
            # An ID is consumed once even if somebody edits its file afterwards.
            previous = next((r for r in reversed(store.owned('idle_worklist'))
                if r.get('resume_request_id') == doc['request_id'] and r['phase'] not in ('deferred', 'queued')), None)
            if previous:
                continue
            return dict(task='rl_experiment', task_kind='rl_experiment', version='resume-'+digest,
                        resume_request_id=doc['request_id'], resume_request_path=str(path),
                        resume_request=doc, rerun_mark='resume:'+doc['request_id'],
                        recipient=doc['recipient'], no_email=doc['no_email'],
                        revisions={'rl_resume': {'sha256': digest, 'source_commit': doc['source_commit']}})
        except (ValueError, KeyError, TypeError) as exc:
            store.event(dict(id='rl-resume-config-'+digest, owner='idle_worklist', task_kind='rl_experiment'),
                        'configuration_error', f'Invalid resume request {path}: {exc}')
    return None


def recheck(request, root):
    path = Path(request['resume_request_path'])
    if path.parent.resolve() != request_dir(root).resolve():
        raise ValueError('Resume request path is outside the host queue')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != request['revisions']['rl_resume']['sha256']:
        raise ValueError('Queued resume request changed')
    doc = validate_request(json.loads(raw), root)
    for name, key in [('config.json', 'config_sha256'), ('identity.json', 'identity_sha256')]:
        if hashlib.sha256((Path(doc['run_directory'])/name).read_bytes()).hexdigest() != doc[key]:
            raise ValueError(f'Original {name} changed after the resume request')
    if json.loads((Path(doc['run_directory'])/'system.json').read_text()).get('git_revision') != doc['source_commit']:
        raise ValueError('Original source commit changed after the resume request')
    return doc
