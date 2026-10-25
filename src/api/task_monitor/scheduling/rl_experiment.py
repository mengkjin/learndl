"""Lightweight RL configuration and Git eligibility; never import training here."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager, ExitStack
from pathlib import Path

import portalocker
import yaml

from .config import UniqueLoader, installed_config, runtime_dir

CONFIG_PATH = 'configs/strategy/rl_experiment.yaml'
TASK = 'rl_experiment'


class ConfigurationPending(ValueError):
    """The checkout is not yet a usable committed configuration."""


def parse_config(content: str, *, yaml_format: bool = True) -> dict:
    doc = yaml.load(content, Loader=UniqueLoader) if yaml_format else json.loads(content)
    if not isinstance(doc, dict):
        raise ValueError('RL configuration must be a mapping')
    if not yaml_format:  # Legacy experiment JSON, not a scheduling envelope.
        mode = 'walk_forward' if doc.get('kind') == 'walk_forward' else 'single'
        result = dict(enabled=True, rerun_mark=None, mode=mode, no_email=False,
                      recipient=doc.get('recipient'), experiment=doc)
    else:
        extra = set(doc) - {'enabled', 'rerun_mark', 'mode', 'recipient', 'no_email', 'single', 'walk_forward'}
        if extra:
            raise ValueError(f'Unknown RL scheduling keys: {sorted(extra)}')
        mode = doc.get('mode', 'single')
        if mode not in {'single', 'walk_forward'}:
            raise ValueError('mode must be single or walk_forward')
        result = dict(enabled=doc.get('enabled', True), rerun_mark=doc.get('rerun_mark'),
                      mode=mode, recipient=doc.get('recipient'), no_email=doc.get('no_email', False),
                      experiment=doc.get(mode))
    for key in ('enabled', 'no_email'):
        if type(result[key]) is not bool:
            raise ValueError(f'{key} must be a YAML boolean')
    if isinstance(result['rerun_mark'], (dict, list)):
        raise ValueError('rerun_mark must be a scalar')
    if result['recipient'] is not None and not isinstance(result['recipient'], str):
        raise ValueError('recipient must be a string or null')
    config = result['experiment']
    if not isinstance(config, dict):
        raise ValueError(f'{mode} must contain the experiment parameters')
    config = dict(config)
    required = (('alpha', 'start', 'end', 'snapshot_dir', 'output_root') if mode == 'single'
                else ('panel', 'output', 'screen', 'first_quarter', 'last_quarter'))
    if any(config.get(key) in (None, '') for key in required):
        raise ValueError(f'{mode} requires {required}')
    if not isinstance(config.get('training', {}), dict):
        raise ValueError('training must be a mapping')
    if mode == 'walk_forward':
        config['kind'] = 'walk_forward'
    elif config.get('kind') == 'walk_forward':
        raise ValueError('single configuration cannot use kind=walk_forward')
    config['recipient'] = result['recipient']
    result['experiment'] = config
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('RL parameters must contain finite JSON-compatible values; quote date strings') from exc
    return result


def read_config(path: str | Path) -> dict:
    path = Path(path)
    return parse_config(path.read_text(), yaml_format=path.suffix.lower() in {'.yaml', '.yml'})


def settings() -> dict:
    return (installed_config() or {}).get('watchdog', {}).get('jobs', {}).get('rl_experiment', {})


def git_revision(root: Path) -> dict | None:
    """Committed bytes only. The last file commit distinguishes A -> B -> A."""
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout
    if not (root / CONFIG_PATH).exists():
        return None
    if git('status', '--porcelain', '--', CONFIG_PATH).strip():
        raise ConfigurationPending('RL configuration has local/uncommitted changes; waiting for a clean committed file')
    head = git('rev-parse', 'HEAD').strip()
    content = git('show', f'{head}:{CONFIG_PATH}')
    return dict(content=content, git_commit=git('log', '-1', '--format=%H', head, '--', CONFIG_PATH).strip(),
                sha256=hashlib.sha256(content.encode()).hexdigest())


def observe(store, *, root: Path | None = None, throttle: bool = False) -> dict | None:
    """Persist the latest pending version without creating a training attempt."""
    from src.proj import PATH
    root = root or PATH.main
    state = dict(status='disabled', updated_at=time.time())
    old = status(store)
    request = None
    try:
        options = settings()
        if options.get('enabled'):
            # The two watchdog jobs share an arbiter, but RL's configured
            # polling interval still applies when the model job runs faster.
            # Workers always bypass this cache before claiming/starting work.
            cached = (throttle and '_revision' in old and
                      time.time() - old.get('observed_at', 0) < options.get('interval_seconds', 120))
            revision = old['_revision'] if cached else git_revision(root)
            state.update(_revision=revision, observed_at=old['observed_at'] if cached else time.time())
            state['status'] = 'configuration missing'
            if revision:
                identity = {key: revision[key] for key in ('git_commit', 'sha256')}
                version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                state.update(version=version, **identity)
                # Respect notification preferences even if another field fails validation.
                raw = yaml.load(revision['content'], Loader=UniqueLoader)
                state['no_email'] = isinstance(raw, dict) and raw.get('no_email') is True
                if isinstance(raw, dict) and isinstance(raw.get('recipient'), str):
                    state['recipient'] = raw['recipient']
                config = parse_config(revision['content'])
                state.update(mode=config['mode'], rerun_mark=config['rerun_mark'],
                             no_email=config['no_email'], status='disabled')
                if config['enabled']:
                    previous = next((r for r in reversed(store.owned('idle_worklist'))
                                     if r.get('task_kind') == 'rl_experiment' and r['version'] == version
                                     and r['phase'] != 'deferred'), None)
                    state['status'] = previous['phase'] if previous else 'pending'
                    if previous:
                        state['run_id'] = previous['id']
                        from .recovery import evidence
                        details = evidence(store, previous['id'])
                        state.update({k: previous.get(k) for k in ('started_at', 'finished_at', 'returncode', 'log_path')})
                        state.update({k: details.get(k) for k in ('output_directory', 'error')})
                    else:
                        request = dict(task=TASK, task_kind=TASK, version=version,
                                       revisions={'rl_config': revision}, config=config,
                                       rerun_mark=config['rerun_mark'], no_email=config['no_email'])
    except ConfigurationPending as exc:
        state.update(status='waiting for committed configuration', error=str(exc))
    except (ValueError, OSError, subprocess.SubprocessError, yaml.YAMLError) as exc:
        state.update(status='configuration error', error=f'{type(exc).__name__}: {exc}')
        key = hashlib.sha256(json.dumps({k: v for k, v in state.items() if k not in {'updated_at', 'observed_at'}},
                                       sort_keys=True, default=str).encode()).hexdigest()
        if not state.get('no_email'):
            store.event(dict(id=f'rl-config-{key}', owner='idle_worklist', task_kind=TASK,
                             recipient=state.get('recipient')),
                        'configuration_error', state['error'])
    if state.get('status') == old.get('status') == 'pending' and state.get('version') == old.get('version'):
        state['wait_reason'] = old.get('wait_reason')
    save_state(store, state)
    return request


def save_state(store, state: dict) -> None:
    with store.connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS rl_status (id INTEGER PRIMARY KEY, data TEXT NOT NULL)')
        row = db.execute('SELECT data FROM rl_status WHERE id=1').fetchone()
        old = json.loads(row[0]) if row else {}
        if any(old.get(k) != state.get(k) for k in ('version', 'status', 'error', 'wait_reason')):
            print(f'[RL scheduling] {state.get("status")}: {state.get("wait_reason") or state.get("error") or state.get("version", "")}',
                  flush=True)
        db.execute('INSERT OR REPLACE INTO rl_status VALUES (1, ?)', (json.dumps(state),))


def status(store) -> dict:
    with store.connect() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='rl_status'").fetchone():
            return {}
        row = db.execute('SELECT data FROM rl_status WHERE id=1').fetchone()
    return json.loads(row[0]) if row else {}


def waiting(store, reason: str) -> None:
    state = status(store)
    if state.get('status') == 'pending':
        save_state(store, dict(state, wait_reason=reason))


@contextmanager
def fit_guard():
    """Acquire occupancy exclusively to close admission races with shared fits."""
    from src.proj.util.script import FitLock, FitLockNN
    with ExitStack() as stack:
        for path in (FitLock.lock_path(), FitLockNN.lock_path()):
            path.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(portalocker.Lock(str(path), mode='a', timeout=0,
                                                flags=portalocker.LOCK_EX | portalocker.LOCK_NB))
        yield


def execute(request: dict, store) -> int:
    """Called only inside the isolated worker child, with a committed snapshot."""
    from src.proj import PATH
    from src.proj.util.cli.script_session import as_script_main
    from src.proj.util.filesys.dynamic_import import dynamic_modules
    from .recovery import record
    with ExitStack() as guards:
        try:
            guards.enter_context(fit_guard())
        except portalocker.LockException:
            return 75
        # Training exceptions (including lock errors inside the experiment) are
        # real failed attempts, never eligibility deferrals.
        try:
            current = git_revision(PATH.main)
        except ConfigurationPending:
            return 75
        expected = request['revisions']['rl_config']
        if (not settings().get('enabled') or current != expected):
            return 75
        folder = runtime_dir() / 'rl_experiment' / request['id']
        folder.mkdir(parents=True, exist_ok=True)
        config = dict(request['config']['experiment'])
        output_key = 'output' if request['config']['mode'] == 'walk_forward' else 'output_root'
        config[output_key] = str(Path(config[output_key]).expanduser().resolve() / 'automatic' / request['id'])
        config['scheduling_provenance'] = dict(run_id=request['id'], version=request['version'],
                                              revision=expected, rerun_mark=request.get('rerun_mark'))
        (folder / 'rl_experiment.yaml').write_text(expected['content'])
        resolved = folder / 'experiment.json'
        resolved.write_text(json.dumps(config, ensure_ascii=False, indent=2))
        record(store, request['id'], output_directory=config[output_key], config_path=str(resolved))
        script = PATH.scpt / '4_train' / '3_train_rl_experiment.py'
        # BackendTaskRecorder parses sys.argv even for Python keyword calls.
        # Never expose the worker's `_child RUN_ID` arguments to ScriptTool.
        previous_argv = sys.argv
        sys.argv = [str(script)]
        try:
            with as_script_main(script):
                module = next(iter(dynamic_modules(script)))
                task = module.main(config_path=str(resolved), recipient=request['config']['recipient'],
                                   no_email=request['no_email'], email=not request['no_email'])
        finally:
            sys.argv = previous_argv
        success = bool(task.success and task.execution_success)
        record(store, request['id'], script_outcome='success' if success else 'error',
               error=None if success else 'RL script failed; see its task log and diagnostic bundle')
        return 0 if success else 1
