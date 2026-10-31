"""Resume an existing suite using its exact, verified historical RL sources.

No fingerprint bypass: old code is exported from local Git into the attempt
directory. Only a dedicated worker process may swap the RL package namespace.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import itertools
import json
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from .bundle import _sha256
from .progress import progress, quiet_output
from .recover_results import read_json
from src.proj.util.web.task_mail import save_json

PREFIX = 'src.res.rl_experimental'
SOURCE = 'src/res/rl_experimental'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def git_sources(root, revision):
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40,64}', revision):
        raise ValueError('Original system.json must identify a full Git commit')
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True,
                              capture_output=True, timeout=30).stdout
    names = git('ls-tree', '--name-only', revision, SOURCE+'/').decode().splitlines()
    files = {Path(name).name: git('show', f'{revision}:{name}') for name in names
             if Path(name).parent.as_posix() == SOURCE and name.endswith('.py')}
    if not {'__init__.py', 'walk_forward.py', 'training.py'} <= files.keys():
        raise ValueError('Historical RL sources unavailable in local Git')
    return files


def inspect_resume(run, root):
    """Distinguish missing work from corrupt completed work before enqueueing."""
    run, root = Path(run).resolve(), Path(root).resolve()
    config = read_json(run/'config.json')
    if Path(config['output']).resolve() != run:
        raise ValueError('Original output path does not match the requested run')
    identity, system, plan = [read_json(run/name) for name in ('identity.json', 'system.json', 'plan.json')]
    files = git_sources(root, system['git_revision'])
    code = fingerprint({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    identity_config = {k: v for k, v in config.items() if k not in ('stage', 'recipient', 'bundle_max_mb', 'profile_report')}
    expected = fingerprint({'config': identity_config, 'panel': identity['panel_contract'], 'code': code})
    if expected != identity['fingerprint']:
        raise ValueError('Original config/data/source fingerprint mismatch; refusing to mix experiments')
    completed, pending, versions = {}, [], []
    tasks = [(run/'screen'/f'p{p:g}-b{b:g}-s{s}', config['screen_steps'], s)
             for p, b, s in itertools.product(config['penalties'], config['bonuses'], config['seeds'])]
    tasks += [(run/'quarters'/f['name']/a/f'seed-{s}', config['quarter_steps'], s)
              for f, a, s in itertools.product(plan['folds'], ('original', 'penalty', 'retention'), config['seeds'])]
    if len(tasks) != plan['total_training_tasks']:
        raise ValueError('Task plan/config mismatch')
    selected = run/'selected.json'
    if selected.exists():
        if _sha256(selected) != read_json(run/'selection_fingerprint.json')['sha256']:
            raise ValueError('Frozen selection was modified')
    elif any((run/'quarters').glob('*/**/task_status.json')):
        raise ValueError('Quarter training exists without its frozen selection')
    for folder, budget, seed in tasks:
        status_path = folder/'task_status.json'
        status = read_json(status_path) if status_path.exists() else {}
        key = folder.relative_to(run).as_posix()
        if status.get('status') != 'success':
            # Do not overwrite a possibly completed but uncommitted task.
            if (folder/'metrics.json').exists():
                raise ValueError(f'Ambiguous completion (metrics without success receipt): {key}')
            pending.append(key)
            continue
        metrics = read_json(folder/'metrics.json')
        task_hash = fingerprint({'contract': metrics['panel_contract'], 'splits': metrics['splits'],
                                 'config': metrics['config'], 'implementation': code})
        if task_hash != status['fingerprint']:
            raise ValueError(f'Task fingerprint mismatch: {key}')
        if metrics['actual_timesteps'] < budget or metrics['config']['seed'] != seed or metrics['config']['total_timesteps'] != budget:
            raise ValueError(f'Task seed/budget mismatch: {key}')
        for name in ('metrics.json', 'normalizer.npz', 'best_model.zip', 'final_model.zip'):
            if _sha256(folder/name) != status['artifacts'][name]:
                raise ValueError(f'Completed artifact corrupted: {key}/{name}')
        if not (folder/'validation_baseline_metrics.json').is_file():
            raise ValueError(f'Missing completed validation baseline: {key}')
        completed[key] = dict(status['artifacts'])
        versions.append(metrics['versions'])
    if selected.exists() and any(key.startswith('screen/') for key in pending):
        raise ValueError('Frozen selection exists but screening artifacts are missing; refusing to change completed selection')
    # Check dependencies without loading Torch or requesting a GPU in autofix.
    for version in versions:
        for package in ('numpy', 'torch', 'gymnasium', 'stable_baselines3'):
            installed, recorded = importlib.metadata.version(package), version[package]
            # PyPI torch metadata may omit the +cuXXX suffix exposed by
            # torch.__version__. Check that full build in the GPU worker below.
            if package == 'torch':
                installed, recorded = installed.split('+')[0], recorded.split('+')[0]
            if installed != recorded:
                raise ValueError(f'{package} version changed; restore the original environment before resuming')
    return {'run': str(run), 'source_commit': system['git_revision'], 'code_fingerprint': code,
            'config_sha256': _sha256(run/'config.json'), 'identity_sha256': _sha256(run/'identity.json'),
            'completed': completed, 'pending': pending,
            'stage': 'quarters' if selected.exists() and all(not x.startswith('screen/') for x in pending) else 'all',
            'recipient': config.get('recipient')}


@contextmanager
def historical_package(folder):
    """Worker-only namespace isolation so checkpoint/reward imports use old code."""
    saved = {k: v for k, v in sys.modules.items() if k == PREFIX or k.startswith(PREFIX+'.')}
    parent = importlib.import_module('src.res')
    old_attribute = getattr(parent, 'rl_experimental', None)
    for key in saved:
        del sys.modules[key]
    try:
        spec = importlib.util.spec_from_file_location(PREFIX, folder/'__init__.py', submodule_search_locations=[str(folder)])
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load historical package from {folder}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[PREFIX] = module
        setattr(parent, 'rl_experimental', module)
        spec.loader.exec_module(module)
        yield importlib.import_module(PREFIX+'.walk_forward')
    finally:
        for key in list(sys.modules):
            if key == PREFIX or key.startswith(PREFIX+'.'):
                del sys.modules[key]
        sys.modules.update(saved)
        setattr(parent, 'rl_experimental', old_attribute)


def resume_suite(run, root, attempt):
    """Finish missing tasks; completed tasks must remain byte-for-byte untouched."""
    run, root, attempt = Path(run).resolve(), Path(root).resolve(), Path(attempt).resolve()
    plan = inspect_resume(run, root)
    attempt.mkdir(parents=True, exist_ok=False)
    save_json(attempt/'resume_plan.json', plan)
    for name in ('config.json', 'system.json', 'status.json', 'identity.json', 'selected.json', 'selection_fingerprint.json'):
        if (run/name).exists():
            shutil.copy2(run/name, attempt/name)
    source = attempt/'source'
    source.mkdir()
    for name, data in git_sources(root, plan['source_commit']).items():
        (source/name).write_bytes(data)
    baseline_system = read_json(run/'system.json')
    try:
        with quiet_output(attempt/'detail.log'), historical_package(source) as engine:
            import torch
            if str(torch.__version__) != baseline_system['torch']:
                raise ValueError('Torch runtime/build changed; restore the original environment before resuming')
            # The legacy engine still enforces both suite and per-task hashes,
            # including a full panel contract after loading the snapshot.
            if engine.code_fingerprint() != plan['code_fingerprint']:
                raise ValueError('Exported source fingerprint mismatch')
            experiment = importlib.import_module(PREFIX+'.experiment')
            setattr(experiment, '_system_info', lambda: baseline_system)
            phase = {'quarter': None, 'replay': None}
            original_progress = engine.progress

            def visible_progress(stage, message, **kwargs):
                original_progress(stage, message, **kwargs)
                if kwargs.get('milestone'):
                    progress(stage, message, milestone=True, warning=kwargs.get('warning', False))
                if hasattr(engine, '_run_walk_forward'):
                    return
                if stage in phase:
                    quarter = message.split('/')[0]
                    if phase[stage] and phase[stage] != quarter:
                        progress('resume', f'{phase[stage]}: {stage} complete', milestone=True)
                    if stage == 'replay' and phase['quarter']:
                        progress('resume', f'{phase["quarter"]}: training complete', milestone=True)
                        phase['quarter'] = None
                    phase[stage] = quarter

            setattr(engine, 'progress', visible_progress)
            progress('resume', f'Reuse {len(plan["completed"])} tasks; finish {len(plan["pending"])}; source={plan["source_commit"]}', milestone=True)
            engine.run_walk_forward(run/'config.json', stage=plan['stage'])
            if phase['replay']:
                progress('resume', f'{phase["replay"]}: test replay complete', milestone=True)
        for folder, artifacts in plan['completed'].items():
            for name, digest in artifacts.items():
                if _sha256(run/folder/name) != digest:
                    raise ValueError(f'Resumption changed a completed artifact: {folder}/{name}')
        save_json(attempt/'outcome.json', {'status': 'success', 'reused': len(plan['completed']), 'trained': len(plan['pending'])})
        return plan
    except BaseException as exc:
        save_json(attempt/'outcome.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
