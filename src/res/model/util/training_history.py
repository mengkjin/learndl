"""Machine-local training provenance, independent of worklist completion policy."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from importlib.metadata import PackageNotFoundError, version
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any
from collections.abc import Callable

import psutil

from src.proj import MACHINE, PATH, Save

__all__ = ['TrainingRun', 'collect_training_runs', 'record_training', 'file_revision', 'write_json']

_OBSERVER: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar('training_observer', default=None)
_RUNS: ContextVar[list[dict[str, Any]] | None] = ContextVar('training_runs', default=None)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ['git', '-C', str(PATH.main), *args], stderr=subprocess.DEVNULL,
            text=True, timeout=15,
        ).rstrip('\n')
    except (OSError, subprocess.SubprocessError):
        return None


def file_revision(path: Path) -> dict[str, Any]:
    """Raw bytes plus last file commit: unrelated commits never invalidate a file."""
    content = path.read_bytes()
    try:
        relative = str(path.resolve().relative_to(PATH.main.resolve()))
    except ValueError:
        relative = str(path)
    return {
        'path': str(path.resolve()), 'sha256': hashlib.sha256(content).hexdigest(),
        'git_commit': git('log', '-1', '--format=%H', '--', relative),
        'content': content.decode('utf-8'),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic replacement so interruption cannot leave a half-written record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('w') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def collect_training_runs(on_update: Callable[[dict[str, Any]], None] | None = None):
    """Link worklist attempts to general runs without making history depend on it."""
    runs: list[dict[str, Any]] = []
    token = _RUNS.set(runs)
    observer_token = _OBSERVER.set(on_update)
    try:
        yield runs
    finally:
        _RUNS.reset(token)
        _OBSERVER.reset(observer_token)


class TrainingRun:
    """One pipeline invocation, including setup failures and test-only runs."""

    def __init__(self, trainer):
        self.root = PATH.lc_machine / 'training_history'
        self.run_id = uuid.uuid4().hex
        self.folder = self.root / self.run_id
        self.path = self.folder / 'run.json'
        self.folder.mkdir(parents=True, exist_ok=True)
        status = git('status', '--porcelain=v1', '--untracked-files=all')
        patch = git('diff', '--binary', 'HEAD', '--', 'src', 'scripts', 'configs', 'pyproject.toml', 'uv.lock')
        if patch:
            (self.folder / 'working_tree.patch').write_text(patch + '\n')
        packages = {}
        for package in ('torch', 'numpy', 'pandas', 'lightgbm', 'xgboost', 'catboost'):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                pass
        self.data: dict[str, Any] = {
            'schema_version': 1, 'run_id': self.run_id, 'status': 'running',
            'started_at': now(), 'finished_at': None, 'machine': MACHINE.name,
            'hostname': platform.node(), 'pid': os.getpid(),
            'process_created': psutil.Process().create_time(),
            'python': sys.version, 'packages': packages, 'argv': sys.argv, 'cwd': os.getcwd(),
            'requested': json.loads(json.dumps(trainer._config_kwargs | trainer.input_model_kwargs, default=str)),
            'model_path': None,
            'git': {'head': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
                    'branch': git('branch', '--show-current'), 'repo': str(PATH.main.resolve()),
                    'commit_time': git('show', '-s', '--format=%cI', 'HEAD'),
                    'subject': git('show', '-s', '--format=%s', 'HEAD'), 'status': status,
                    'dirty': bool(status) if status is not None else None,
                    'patch': 'working_tree.patch' if patch else None},
        }
        self.save()
        collected = _RUNS.get()
        if collected is not None:
            collected.append(self.data)
        schedule_name = trainer._config_kwargs.get('schedule_name')
        if schedule_name:
            from src.res.model.util.config.config import ScheduleConfig
            try:
                source = ScheduleConfig.find_path(name=schedule_name)
                if source:
                    self.data['schedule_source'] = file_revision(source)
            except (OSError, AssertionError) as exc:
                self.data['schedule_source_error'] = str(exc)
        self.save()

    def save(self):
        write_json(self.path, self.data)
        if observer := _OBSERVER.get():
            observer(self.data)

    def configured(self, config):
        """Capture effective values, including overrides and archived resume config."""
        self.data.update(json.loads(json.dumps({
            'model_path': str(config.base_path.base.resolve()),
            'model_name': config.model_name, 'schedule_name': config.model_config.schedule_name,
            'stages': list(config.queue_of_stages), 'resume': config.is_resuming,
            'kind': 'training' if 'fit' in config.queue_of_stages else 'evaluation',
            'short_test': config.short_test,
            'model_config': dict(config.model_config.Param),
            'schedule_config': dict(config.schedule_config.Param),
            'algo_config': dict(config.algo_config.Param),
            'boost_head_config': dict(config.boost_head_config.Param) if config.boost_head_config else None,
        }, default=str)))
        self.save()

    def finish(self, status: str, error: BaseException | None = None):
        self.data.update(status=status, finished_at=now(), git_end=git('rev-parse', 'HEAD'))
        if error is not None:
            self.data['error'] = f'{type(error).__name__}: {error}'
        self.save()

    @classmethod
    def read(cls, path: Path) -> dict[str, Any]:
        """Report killed/rebooted processes as interrupted, never as successful."""
        data = json.loads(path.read_text())
        if data['status'] == 'running':
            try:
                process = psutil.Process(data['pid'])
                alive = (process.create_time() == data['process_created'] and
                         process.status() != psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                alive = False
            except psutil.AccessDenied:
                alive = True
            if not alive:
                data.update(status='interrupted', detected_at=now())
                write_json(path, data)
        return data


def record_training(func):
    """Record the shared training lifecycle; failure must propagate to ScriptTool."""
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        run = TrainingRun(self)
        self.training_run = run
        try:
            result = func(self, *args, **kwargs)
            Save.async_wait_all(caller_name='training_history')
        except BaseException as exc:
            run.finish('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed', exc)
            raise
        else:
            run.finish('success')
            return result
        finally:
            self.training_run = None
    return wrapped
