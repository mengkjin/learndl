"""Versioned migrations for host-local project files."""
from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Collection
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
import types
from typing import Literal, TypedDict, cast
import uuid

from src.automation.console import logger
from src.automation.paths import ROOT
from src.automation.patch_diagnostics import failure_summary

__all__ = [
    'AttemptRecord',
    'CodeBusy',
    'Context',
    'MigrationError',
    'MigrationStatus',
    'activation',
    'admission_reason',
    'apply_pending',
    'business_lease',
    'code_lease',
    'initialize',
    'queue_completed_batches',
    'load_patch_support',
    'main',
    'needs_migration',
    'status',
]

MIN_DATA_VERSION = '0.7.1'
VERSION_FILE = 'src/__version__.py'
PATCH_DIRECTORY = 'scripts/.version_patch'
_LEGACY_PATCH_DIRECTORY = 'scripts/.version_autofix'
DEFAULT_TIMEOUT = 1800
VERSION = re.compile(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)')
FILENAME = re.compile(r'([0-9]{3,})_([a-z][a-z0-9_]*)\.py')


class MigrationError(RuntimeError):
    """The migration cannot safely advance."""


class CodeBusy(MigrationError):
    """Code is occupied or the installed data version is not ready for new work."""


def version_key(version: str) -> tuple[int, ...]:
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        raise MigrationError(f'Invalid project version: {version!r}')
    return tuple(map(int, version.split('.')))


def code_version(root: Path) -> str:
    return parse_code_version((root / VERSION_FILE).read_bytes())


def parse_code_version(source: str | bytes) -> str:
    body = ast.parse(source).body
    if (len(body) != 1 or not isinstance(body[0], ast.Assign) or len(body[0].targets) != 1
            or not isinstance(body[0].targets[0], ast.Name)
            or body[0].targets[0].id != '__version__' or not isinstance(body[0].value, ast.Constant)):
        raise MigrationError(f'Expected one literal __version__ assignment in {VERSION_FILE}')
    version = body[0].value.value
    if not isinstance(version, str):
        raise MigrationError(f'Expected a string version in {VERSION_FILE}')
    version_key(version)
    return version


@dataclass(frozen=True)
class Patch:
    id: str
    version: str
    number: int
    relative_path: str
    digest: str
    timeout: int


@dataclass(frozen=True)
class Context:
    """Host paths and version range available to a patch."""

    root: Path
    source_version: str
    target_version: str
    log_path: Path
    backup_dir: Path


class AttemptRecord(TypedDict):
    """One durable attempt loaded from the migration ledger."""

    id: str
    patch: str
    path: str
    digest: str
    commit_hash: str
    host: str
    attempt: int
    started: float
    finished: float | None
    status: Literal['running', 'success', 'failed', 'interrupted']
    error: str | None
    log_path: str


class MigrationStatus(TypedDict):
    """Snapshot of code, data and patch progress on one checkout."""

    target: str
    baseline: str | None
    applied: str
    pending: list[Patch]
    records: list[AttemptRecord]
    failed: AttemptRecord | None
    ready: bool


def discover(root: Path, target: str) -> dict[str, Patch]:
    directory = root / PATCH_DIRECTORY
    if directory.is_symlink() or directory.parent.is_symlink():
        raise MigrationError(f'Symlink is not allowed: {PATCH_DIRECTORY}')
    if not directory.exists():
        return {}
    patches = []
    for folder in directory.iterdir():
        if not VERSION.fullmatch(folder.name):
            continue
        if folder.is_symlink() or not folder.is_dir():
            raise MigrationError(f'Expected a version directory: {folder.name}')
        if version_key(folder.name) > version_key(target):
            continue
        manifest = folder / 'timeouts.json'
        if manifest.is_symlink():
            raise MigrationError(f'Symlink is not allowed: {manifest.relative_to(root)}')
        timeouts = json.loads(manifest.read_text()) if manifest.exists() else {}
        if not isinstance(timeouts, dict) or any(type(value) is not int or value < 1 for value in timeouts.values()):
            raise MigrationError(f'Expected positive integer seconds in {manifest.relative_to(root)}')
        seen = set()
        names = set()
        for path in folder.glob('*.py'):
            match = FILENAME.fullmatch(path.name)
            if not match or path.is_symlink() or not path.is_file():
                raise MigrationError(f'Invalid patch file: {path.relative_to(root)}')
            number = int(match[1])
            if number in seen:
                raise MigrationError(f'Duplicate patch number {number} in {folder.name}')
            seen.add(number)
            names.add(path.name)
            relative = path.relative_to(root).as_posix()
            patches.append(Patch(f'{folder.name}/{number:03d}', folder.name, number, relative,
                                 hashlib.sha256(path.read_bytes()).hexdigest(), timeouts.get(path.name, DEFAULT_TIMEOUT)))
        if timeouts.keys() - names:
            raise MigrationError(f'Unknown patch in {manifest.relative_to(root)}')
    return {patch.id: patch for patch in sorted(patches, key=lambda item: (version_key(item.version), item.number))}


def ledger_path(runtime: Path) -> Path:
    current = runtime / 'version_patch'
    legacy = runtime / 'version_autofix'
    if legacy.exists() or legacy.is_symlink():
        if current.exists() or current.is_symlink():
            raise MigrationError('Both version patch state directories exist; restore a single authoritative ledger')
        return legacy / 'ledger.sqlite'
    return current / 'ledger.sqlite'


def connect(runtime: Path, *, readonly: bool = False) -> sqlite3.Connection:
    path = ledger_path(runtime)
    db = sqlite3.connect(path.as_uri() + ('?mode=ro' if readonly else '?mode=rw'), uri=True)
    db.row_factory = sqlite3.Row
    return db


def initialize(runtime: Path, *, required: bool = False) -> None:
    """Create only on first installation, under the code and maintenance locks."""
    path = ledger_path(runtime)
    if not path.parent.exists() and not required:
        path.parent.mkdir(parents=True, mode=0o700)
        temp = path.with_suffix('.new')
        with closing(sqlite3.connect(temp)) as db, db:
            db.executescript("""
                CREATE TABLE state (
                    id INTEGER PRIMARY KEY CHECK(id = 1), baseline TEXT, applied TEXT
                );
                INSERT INTO state VALUES (1, NULL, NULL);
                CREATE TABLE attempts (
                    id TEXT PRIMARY KEY, patch TEXT NOT NULL, path TEXT NOT NULL,
                    digest TEXT NOT NULL, commit_hash TEXT NOT NULL, host TEXT NOT NULL,
                    attempt INTEGER NOT NULL, started REAL NOT NULL, finished REAL,
                    status TEXT NOT NULL CHECK(status IN ('running','success','failed','interrupted')),
                    error TEXT, log_path TEXT NOT NULL,
                    UNIQUE(patch, attempt)
                );
                CREATE UNIQUE INDEX successful_patch ON attempts(patch) WHERE status='success';
            """)
        temp.replace(path)
    with closing(connect(runtime, readonly=True)) as db:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise MigrationError('Version migration ledger is corrupt; restore from backup')
        read_state(db)
        db.execute('SELECT id, patch, path, digest, commit_hash, host, attempt, started, finished, status, error, log_path FROM attempts LIMIT 0')


def queue_completed_batches(runtime: Path, enqueue: Callable[[str, str], None]) -> None:
    """Transfer committed batch notices to the mail outbox before acknowledging them."""
    if not ledger_path(runtime).is_file():
        return
    with closing(connect(runtime, readonly=True)) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='completed_batches'").fetchone() is None:
            return
        pending = db.execute('SELECT id, data FROM completed_batches WHERE queued=0 ORDER BY rowid').fetchall()
    if not pending:
        return
    for notice in pending:
        enqueue(notice['id'], notice['data'])
    with closing(connect(runtime)) as db, db:
        db.executemany('UPDATE completed_batches SET queued=1 WHERE id=?', [(row['id'],) for row in pending])


def read_state(db: sqlite3.Connection) -> tuple[str | None, str | None]:
    row = db.execute('SELECT baseline, applied FROM state WHERE id=1').fetchone()
    if row is None:
        raise MigrationError('Invalid version migration state')
    baseline, applied = row
    if baseline is None and applied is None:
        return None, None
    if not isinstance(baseline, str) or not isinstance(applied, str):
        raise MigrationError('Invalid version migration state')
    if version_key(applied) < version_key(baseline):
        raise MigrationError('Applied version precedes the baseline')
    return baseline, applied


def _first_payload(directory: Path, *, ignored_files: Collection[Path] = (), receipts_only: bool = False) -> Path | None:
    """Ignore empty scaffolding and harmless checkout metadata."""
    stack = [directory]
    while stack:
        path = stack.pop()
        if path.is_symlink():
            return path
        if (path in ignored_files or path.name == '.DS_Store') and path.is_file():
            continue
        try:
            stack.extend(path.iterdir())
        except FileNotFoundError:
            continue
        except NotADirectoryError:
            if not receipts_only or path.name == 'delivery.json':
                return path
    return None


def _fresh_baseline(root: Path, runtime: Path, target: str) -> str:
    """Allow automatic adoption only before versioned local data is present."""
    roots = {root / name for name in ('data', 'models', 'results')}
    roots.update(runtime.parent / name for name in ('optuna', 'tensorboard'))
    outboxes = {runtime.parent / 'email_outbox'}
    logs = root / '.logs'
    if root.resolve() == ROOT:
        from src.automation.artifact_exchange.core import Config
        from src.proj import PATH

        roots.update((PATH.data, PATH.model, PATH.result, PATH.optuna, PATH.tsboard, Config.load().results))
        outboxes.add(PATH.email_outbox)
        logs = PATH.logs
    ignored_files = {root / 'results' / 'README.md', *(outbox / '.lock' for outbox in outboxes)}
    for directory in sorted(roots | outboxes):
        if payload := _first_payload(directory, ignored_files=ignored_files):
            break
    else:
        payload = _first_payload(logs, receipts_only=True)
    if payload is not None:
        raise MigrationError(
            f'Existing data without a version patch baseline: {payload}. '
            'Restore its original ledger or run python -m src.automation.version_patch '
            'init --baseline <data-version> before applying patches.'
        )
    return target


def _matches_success(patch: Patch, path: str, digest: str) -> bool:
    if path != patch.relative_path:
        return False
    if digest == patch.digest:
        return True
    # These exact revisions leave the data transformations and verification
    # unchanged. They update the open-file probe or temporary-file handling.
    compatible = {
        'scripts/.version_patch/0.7.3/001_factor_residual_snapshot.py': (
            '81ba2996a6f47dfe6329a896fed6e5d7bcb2c9274c7816e378461ae6d6fcd5b1',
            '56f1a6df899a5828af288265bc85f1f3facbaedebaa50732037c310da54c089b',
        ),
        'scripts/.version_patch/0.7.7/001_status_logger.py': (
            '8fcdcac45e7b16adf9c0eeefa5f9f097b50f5787cb5e4256d053ba6ce2cb2020',
            'b68b6b3bec58dbae2541085f28dc8a532c1c237cd8049e64b7628e2644ce8c93',
        ),
    }
    return compatible.get(path) == (digest, patch.digest)


def inspect(root: Path, runtime: Path, db: sqlite3.Connection | None) -> MigrationStatus:
    target = code_version(root)
    baseline, applied = read_state(db) if db is not None else (None, None)
    current = applied if applied is not None else _fresh_baseline(root, runtime, target)
    if version_key(target) < version_key(current):
        raise MigrationError(f'Code version {target} is older than data version {current}')
    patches = discover(root, target)
    records: list[AttemptRecord] = []
    if db is not None:
        records = [cast(AttemptRecord, dict(row)) for row in db.execute('SELECT * FROM attempts ORDER BY patch, attempt')]
    if baseline is None and records:
        raise MigrationError('Version patch attempts exist without a baseline; restore the ledger')
    latest = {row['patch']: row for row in records}
    for row in records:
        if row['status'] == 'success':
            patch = patches.get(row['patch'])
            recorded_path = row['path']
            if recorded_path.startswith(_LEGACY_PATCH_DIRECTORY + '/'):
                recorded_path = PATCH_DIRECTORY + recorded_path.removeprefix(_LEGACY_PATCH_DIRECTORY)
            if patch is None or not _matches_success(patch, recorded_path, row['digest']):
                raise MigrationError(f'Successful patch changed or disappeared: {row["patch"]}')
    pending = [patch for patch in patches.values() if version_key(baseline or current) < version_key(patch.version)
               and (patch.id not in latest or latest[patch.id]['status'] != 'success')]
    if any(version_key(patch.version) <= version_key(current) for patch in pending):
        raise MigrationError('A new patch was added to an already applied version')
    unfinished = [row for row in latest.values() if row['status'] != 'success']
    failed = unfinished[0] if unfinished else None
    if len(unfinished) > 1:
        raise MigrationError('More than one unfinished patch in the ledger')
    if failed and (not pending or pending[0].id != failed['patch']):
        raise MigrationError(f'Unfinished patch changed identity or order: {failed["patch"]}')
    return MigrationStatus(target=target, baseline=baseline, applied=current, pending=pending,
                           records=records, failed=failed,
                           ready=baseline is not None and target == current and failed is None)


def activation(config: dict | None) -> bool:
    return bool(config and config.get('version_patch', config.get('version_autofix')) is True)


def status(root: Path, runtime: Path, *, required: bool = True) -> MigrationStatus:
    """Read a consistent ledger snapshot without creating or modifying it."""
    if not required and not ledger_path(runtime).parent.exists():
        return inspect(root, runtime, None)
    with closing(connect(runtime, readonly=True)) as db:
        db.execute('BEGIN')
        return inspect(root, runtime, db)


_STATUS_ERRORS = (MigrationError, OSError, sqlite3.Error, ValueError, SyntaxError)


def needs_migration(root: Path, runtime: Path) -> bool:
    """Route pending work and unreadable state away from the Git fetch path."""
    try:
        state = status(root, runtime)
    except _STATUS_ERRORS:
        return True
    return not state['ready']


def admission_reason(root: Path, runtime: Path, *, announce_pending: bool = False) -> str | None:
    """Return the reason a new business process must wait, if any."""
    try:
        state = status(root, runtime)
        if state['baseline'] is None:
            return 'Version patch ledger is awaiting automatic initialization'
        if state['ready']:
            return None
        command = 'python -m src.automation.version_patch'
        if attempt := state['failed']:
            action = (f'{command} status' if attempt['status'] == 'running'
                      else f'{command} retry {attempt["patch"]}')
            return f'Version patch {attempt["patch"]} ({attempt["status"]}). See {attempt["log_path"]}. Run {action}'
        # A worker can exit between patches in the same version, after changing
        # data but before recording the new applied version or the next attempt.
        applied = version_key(state['applied'])
        if any(version_key(row['patch'].split('/')[0]) > applied for row in state['records']):
            return f'Version migration partially applied. Run {command} apply'
        committed_catalog(root, 'HEAD', state['target'])
        if announce_pending:
            logger.info('Version migration pending (%s -> %s). Continuing with current data. Details: %s status',
                        state['applied'], state['target'], command, extra={'console_stderr': True})
    except (*_STATUS_ERRORS, subprocess.SubprocessError) as exc:
        return f'Version migration check failed: {exc}'
    return None


@contextmanager
def code_lease(scheduling_runtime: Path, *, shared: bool = False):
    import fcntl

    # Close our copy rather than unlocking: a surviving child retains the lease.
    with (scheduling_runtime / 'git_update.lock').open('a') as stream:
        try:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(stream, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CodeBusy('Code lease occupied') from None
        yield stream.fileno()


@contextmanager
def business_lease(scheduling_runtime: Path, *, root: Path = ROOT):
    """Hold the patch read lock from the ledger check through script completion."""
    scheduling_runtime.mkdir(parents=True, exist_ok=True)
    with code_lease(scheduling_runtime, shared=True) as fd:
        if reason := admission_reason(root, scheduling_runtime.parent):
            raise CodeBusy(reason)
        yield fd


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(root), *args], timeout=15, stderr=subprocess.DEVNULL,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))


def _support_paths(folder: Path) -> list[Path]:
    directory = folder / 'support'
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise MigrationError(f'Expected a regular patch support directory: {directory}')
    paths = sorted(directory.glob('*.py'))
    for path in paths:
        if not path.stem.isidentifier() or path.is_symlink() or not path.is_file():
            raise MigrationError(f'Invalid patch support file: {path}')
    return paths


def load_patch_support(version: str, name: str, *, root: Path = ROOT) -> types.ModuleType:
    """Load a version-local helper on demand, outside normal application code."""
    version_key(version)
    if not name.isidentifier():
        raise MigrationError(f'Invalid patch support name: {name!r}')
    folder = root / PATCH_DIRECTORY / version
    if any(path.is_symlink() for path in (folder, folder.parent, folder.parent.parent)):
        raise MigrationError(f'Expected a regular version directory: {folder}')
    path = folder / 'support' / f'{name}.py'
    if path not in _support_paths(folder):
        raise MigrationError(f'Missing patch support file: {path}')

    key = f'_learndl_patch_{version.replace(".", "_")}_{name}'
    current = sys.modules.get(key)
    if current is not None and getattr(current, '__file__', None) == str(path):
        return current
    spec = spec_from_file_location(key, path)
    if spec is None:
        raise MigrationError(f'Cannot load patch support file: {path}')
    module = module_from_spec(spec)
    sys.modules[key] = module
    try:
        # Use the verified source, even if an older checkout left a bytecode cache.
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return module


def committed_catalog(root: Path, commit: str, target: str) -> None:
    """Do not advance over a patch hidden by a working tree deletion."""
    entries = git(root, 'ls-tree', '-rz', commit, '--', VERSION_FILE, PATCH_DIRECTORY)
    tracked = set()
    expected = {VERSION_FILE}
    for entry in entries.split(b'\0'):
        if not entry:
            continue
        info, raw_name = entry.split(b'\t', 1)
        name = os.fsdecode(raw_name)
        relative = Path(name)
        if name != VERSION_FILE:
            parts = relative.parts
            entrypoint = len(parts) == 4 and (relative.suffix == '.py' or relative.name == 'timeouts.json')
            support = len(parts) == 5 and parts[3] == 'support' and relative.suffix == '.py'
            if (not (entrypoint or support) or not VERSION.fullmatch(parts[2])
                    or version_key(parts[2]) > version_key(target)):
                continue
        if not info.startswith((b'100644 blob ', b'100755 blob ')) or (root / name).is_symlink():
            raise MigrationError(f'Expected a regular committed file: {name}')
        # Admission also checks pending catalogs. Compare the Git blob ID from
        # ls-tree directly to avoid starting a Git process per patch per worker.
        oid = info.split()[2]
        content = (root / name).read_bytes()
        digest = hashlib.new('sha256' if len(oid) == 64 else 'sha1', f'blob {len(content)}\0'.encode())
        digest.update(content)
        if digest.hexdigest().encode() != oid:
            raise MigrationError(f'Patch input differs from committed source: {name}')
        tracked.add(name)
    for patch in discover(root, target).values():
        expected.add(patch.relative_path)
    base = root / PATCH_DIRECTORY
    if base.is_dir():
        for folder in base.iterdir():
            if not VERSION.fullmatch(folder.name) or version_key(folder.name) > version_key(target):
                continue
            manifest = folder / 'timeouts.json'
            if manifest.exists():
                expected.add(manifest.relative_to(root).as_posix())
            expected.update(path.relative_to(root).as_posix() for path in _support_paths(folder))
    if tracked != expected:
        raise MigrationError('The migration catalog differs from the committed checkout')


def check_committed(root: Path, commit: str, patch: Patch) -> None:
    for name in (VERSION_FILE, patch.relative_path, str(Path(patch.relative_path).with_name('timeouts.json'))):
        entry = git(root, 'ls-tree', commit, '--', name)
        path = root / name
        if not entry and not path.exists() and name.endswith('/timeouts.json'):
            continue
        if not entry.startswith((b'100644 blob ', b'100755 blob ')) or path.read_bytes() != git(root, 'show', f'{commit}:{name}'):
            raise MigrationError(f'Patch input differs from committed source: {name}')


def run_patch(root: Path, patch: Patch, context: Context, pass_fds: tuple[int, ...]) -> None:
    payload = json.dumps({key: str(value) for key, value in asdict(context).items()})
    args = [sys.executable, '-m', 'src.automation.version_patch', '_patch',
            str(root / patch.relative_path), patch.digest, str(patch.timeout), payload]
    with context.log_path.open('xb') as log:
        child = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, pass_fds=pass_fds)
        try:
            exit_code = child.wait(timeout=patch.timeout)
            if exit_code:
                raise MigrationError(f'Patch process exited with status {exit_code}')
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                return
            raise MigrationError('Patch left background processes')
        except BaseException as exc:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                except PermissionError:
                    # Some host sandboxes reject group signals even for our own
                    # children. The patch process also enforces its own deadline.
                    try:
                        child.send_signal(sig)
                    except PermissionError:
                        pass
                if sig == signal.SIGTERM:
                    time.sleep(1)
            child.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise MigrationError(f'Patch exceeded its {patch.timeout}s timeout') from None
            raise


def run_child(path: Path, digest: str, timeout: int, payload: str) -> None:
    # Backstop if the supervising process dies while this child owns its lease.
    signal.alarm(timeout + 2)
    context = json.loads(payload)
    for key in ('root', 'log_path', 'backup_dir'):
        context[key] = Path(context[key])
    context = Context(**context)
    os.chdir(context.root)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != digest:
        raise MigrationError('Patch changed before execution')
    module = types.ModuleType('learndl_version_patch')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    apply, verify = module.apply, module.verify
    if not callable(apply) or not callable(verify):
        raise MigrationError('Patch must provide callable apply and verify')
    try:
        apply(context)
    except Exception:
        logger.error('Version patch step: apply')
        raise
    try:
        if verify(context) is not True:
            raise MigrationError('Patch verification did not return True')
    except Exception:
        logger.error('Version patch step: verify')
        raise


def apply_pending(root: Path, runtime: Path, *, pass_fds: tuple[int, ...],
                  notify: Callable[[AttemptRecord], None], retry: str | None = None,
                  baseline: str | None = None) -> dict:
    """Caller holds both exclusive leases for the entire transition."""
    from src.automation.patch_services import PatchServices

    services = PatchServices(root, runtime) if root.resolve() == ROOT else None
    # Recovery is preflight work, not a failed attempt at the next data patch.
    # Do not downgrade a ready journal to blocked while a CLI is registering.
    if services and services.state_data.get('phase') == 'ready':
        try:
            recovery = services.restore(ready=True)
        except Exception as exc:
            return {'status': 'deferred', 'reason': f'Interface recovery pending: {exc}'}
        if services.journal.exists():
            return {'status': 'deferred', 'reason': 'Interface recovery pending', 'interfaces': recovery}
    try:
        result = _apply_pending(root, runtime, pass_fds=pass_fds, notify=notify, retry=retry,
                                baseline=baseline, services=services)
    except BaseException:
        if services and services.journal.exists():
            try:
                services.restore(ready=False)
            except Exception as exc:
                logger.error('Could not recover diagnostic interfaces: %s', exc)
        raise
    if services and result['status'] in {'ready', 'blocked'} and services.journal.exists():
        try:
            result['interfaces'] = services.restore(ready=result['status'] == 'ready')
        except Exception as exc:
            result['interfaces'] = {'errors': {'restore': str(exc)}}
    return result


def _apply_pending(root, runtime, *, pass_fds, notify, retry, baseline, services):
    with closing(connect(runtime)) as db:
        stored_baseline, _ = read_state(db)
        if baseline is not None:
            if stored_baseline is not None or db.execute('SELECT 1 FROM attempts LIMIT 1').fetchone():
                raise MigrationError('Version migration baseline is already initialized')
            if not version_key(MIN_DATA_VERSION) <= version_key(baseline) <= version_key(code_version(root)):
                raise MigrationError(f'Baseline must be between {MIN_DATA_VERSION} and the checked out version')
            with db:
                db.execute('UPDATE state SET baseline=?, applied=? WHERE id=1', (baseline, baseline))
            return {'status': 'initialized', 'baseline': baseline}

        if stored_baseline is None:
            if retry is not None:
                raise MigrationError('retry requires an initialized ledger and a failed patch ID')
            state = inspect(root, runtime, db)
            target = state['target']
            commit = git(root, 'rev-parse', 'HEAD').decode().strip()
            committed_catalog(root, commit, target)
            with db:
                db.execute('UPDATE state SET baseline=?, applied=? WHERE id=1', (target, target))
            return {'status': 'ready', 'version': target, 'patches': []}
        # Obtaining the inherited code lease proves the previous patch is no longer running.
        with db:
            db.execute("UPDATE attempts SET status='interrupted', finished=?, error=? WHERE status='running'",
                       (time.time(), 'Migration worker exited before recording completion'))
        for row in db.execute("SELECT * FROM attempts WHERE status IN ('failed','interrupted')"):
            try:
                notify(cast(AttemptRecord, dict(row)))
            except Exception as exc:
                logger.error('Could not queue version patch email: %s', exc)  # Existing outbox deduplicates by attempt ID.
        state = inspect(root, runtime, db)
        failure = state['failed']
        if retry is not None and (not failure or retry != failure['patch']):
            raise MigrationError('retry requires the first failed or interrupted patch ID')
        if failure and retry is None:
            summary = failure_summary(failure)
            return {'status': 'blocked', 'reason': f'{failure["patch"]} requires manual retry: {summary["error"]}',
                    'failure': summary, 'log': failure['log_path'], 'attempt_id': failure['id']}

        commit = git(root, 'rev-parse', 'HEAD').decode().strip()
        committed_catalog(root, commit, state['target'])
        pending = state['pending']
        source_version = state['applied']
        host = socket.gethostname()
        batch_id = uuid.uuid4().hex
        for index, patch in enumerate(pending):
            started = time.time()
            if git(root, 'rev-parse', 'HEAD').decode().strip() != commit:
                raise MigrationError('Checkout changed during migration')
            key = uuid.uuid4().hex
            folder = ledger_path(runtime).parent / 'attempts' / key
            backups = folder / 'backup'
            backups.mkdir(parents=True, mode=0o700)
            log = folder / 'patch.log'
            with db:
                db.execute("""INSERT INTO attempts
                    (id, patch, path, digest, commit_hash, host, attempt, started, status, log_path)
                    VALUES (?,?,?,?,?,?,(SELECT COUNT(*) + 1 FROM attempts WHERE patch=?),?,'running',?)""",
                           (key, patch.id, patch.relative_path, patch.digest, commit, host,
                            patch.id, started, str(log)))
            context = Context(root, source_version, patch.version, log, backups)
            progress = {'console_stderr': True}
            logger.info('[%d/%d] Applying version patch %s (log: %s)',
                        index + 1, len(pending), patch.id, log, extra=progress)
            try:
                if patch.id == '0.7.2/001' and services is not None:
                    for old in db.execute('SELECT log_path FROM attempts WHERE patch=? AND id!=?',
                                          (patch.id, key)):
                        services.collect_paths(Path(old['log_path']).parent / 'backup')
                run_patch(root, patch, context, pass_fds)
                if patch.id == '0.7.2/001' and services is not None:
                    services.collect_paths(backups)
                if git(root, 'rev-parse', 'HEAD').decode().strip() != commit:
                    raise MigrationError('Checkout changed during patch execution')
                check_committed(root, commit, patch)
            except Exception as exc:
                reason = str(exc)
                if not log.exists():
                    # Setup can fail before the patch subprocess opens its log.
                    try:
                        log.write_text(traceback.format_exc())
                    except OSError as log_error:
                        logger.error('Could not save version patch log: %s', log_error)
                with db:
                    db.execute("UPDATE attempts SET status='failed', finished=?, error=? WHERE id=?",
                               (time.time(), reason, key))
                row = db.execute('SELECT * FROM attempts WHERE id=?', (key,)).fetchone()
                attempt = cast(AttemptRecord, dict(row))
                try:
                    summary = failure_summary(attempt, persist=True)
                except OSError as exc:
                    logger.error('Could not save failure summary: %s', exc)
                    summary = failure_summary(attempt)
                try:
                    notify(attempt)
                except Exception as exc:
                    logger.error('Could not queue version patch email: %s', exc)
                logger.error('[%d/%d] Version patch %s failed: %s', index + 1, len(pending), patch.id, summary['error'])
                return {'status': 'blocked', 'reason': f'{patch.id} failed: {summary["error"]}',
                        'log': str(log), 'failure': summary, 'attempt_id': key}
            finished = time.time()
            last = index + 1 == len(pending)
            with db:
                db.execute("UPDATE attempts SET status='success', finished=? WHERE id=?", (finished, key))
                if last or pending[index + 1].version != patch.version:
                    source_version = state['target'] if last else patch.version
                    db.execute('UPDATE state SET applied=? WHERE id=1', (source_version,))
                if last:
                    # Commit the notice with the final patch, so a worker exit
                    # between migration and SMTP delivery cannot lose the batch.
                    db.execute('CREATE TABLE IF NOT EXISTS completed_batches '
                               '(id TEXT PRIMARY KEY, data TEXT NOT NULL, queued INTEGER NOT NULL DEFAULT 0)')
                    notice = dict(kind='version_patch_success', host=host, commit_hash=commit,
                                  source_version=state['applied'], target_version=state['target'],
                                  patches=[item.id for item in pending],
                                  finished_at=finished, attempts_dir=str(folder.parent))
                    db.execute('INSERT INTO completed_batches(id, data) VALUES (?,?)',
                               (batch_id, json.dumps(notice)))
            logger.info('[%d/%d] Verified version patch %s (%.1fs)',
                        index + 1, len(pending), patch.id, finished - started, extra=progress)
        if not pending:
            if git(root, 'rev-parse', 'HEAD').decode().strip() != commit:
                raise MigrationError('Checkout changed during migration')
            with db:
                db.execute('UPDATE state SET applied=? WHERE id=1', (state['target'],))
        else:
            logger.info('Verified %d version patch(es). Data version %s. Success email recorded for delivery.',
                        len(pending), state['target'], extra={'console_stderr': True})
        return {'status': 'ready', 'version': state['target'], 'patches': [p.id for p in pending]}


def main() -> int:
    if len(sys.argv) == 6 and sys.argv[1] == '_patch':
        run_child(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('status', 'plan', 'apply', 'restore-services'):
        commands.add_parser(command)
    commands.add_parser('retry').add_argument('patch_id')
    commands.add_parser('init').add_argument('--baseline', required=True)
    args = parser.parse_args()
    from src.automation.scheduling.config import installed_config, runtime_dir
    from src.automation.scheduling.git_update import queue_version_failure, training_blocker, version_failure_mail_status
    from src.automation.scheduling.maintenance_gate import MaintenanceBusy, pass_fds, permit

    directory = runtime_dir()
    config = installed_config()
    required = activation(config)
    try:
        result: dict[str, object]
        if args.command in {'status', 'plan'}:
            from src.automation.patch_services import PatchServices

            snapshot = status(ROOT, directory.parent, required=required)
            result = dict(snapshot, pending=[asdict(patch) for patch in snapshot['pending']])
            if snapshot['failed']:
                failure = snapshot['failed']
                result['failure'] = failure_summary(failure)
                result['failure_email'] = version_failure_mail_status(failure['id'])
            if interfaces := PatchServices(ROOT, directory.parent).report():
                result['interfaces'] = interfaces
            if args.command == 'plan':
                del result['records']
        elif args.command == 'restore-services':
            from src.automation.patch_services import PatchServices

            with permit(exclusive=True, blocking=False, task_class='git'), code_lease(directory):
                snapshot = status(ROOT, directory.parent, required=required)
                result = dict(PatchServices(ROOT, directory.parent).restore(ready=snapshot['ready']))
        else:
            directory.mkdir(parents=True, exist_ok=True)
            with permit(exclusive=True, blocking=False, task_class='git'), code_lease(directory) as fd:
                if reason := training_blocker():
                    raise MigrationError(reason)
                initialize(directory.parent, required=required)
                retry = getattr(args, 'patch_id', None)
                if required and args.command != 'init':
                    verify_manual_checkout(ROOT, directory.parent, retry=retry)
                result = apply_pending(ROOT, directory.parent, pass_fds=(fd, *pass_fds()),
                                       notify=queue_version_failure, retry=retry,
                                       baseline=getattr(args, 'baseline', None))
        if isinstance(attempt_id := result.get('attempt_id'), str):
            result['failure_email'] = version_failure_mail_status(attempt_id)
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        interfaces = result.get('interfaces')
        return int(result.get('status') == 'blocked' or bool(result.get('errors'))
                   or (isinstance(interfaces, dict) and bool(interfaces.get('errors'))))
    except (MigrationError, MaintenanceBusy, OSError, ValueError, SyntaxError, sqlite3.Error, subprocess.SubprocessError) as exc:
        logger.error('Version migration: %s', exc)
        return 1


def verify_manual_checkout(root: Path, runtime: Path, *, retry: str | None) -> None:
    """Allow an explicit patch retry to adopt a verified checkout's descendant."""
    from src.automation.scheduling import priority
    from src.automation.scheduling.autofix_store import Store, state_dir
    from src.automation.scheduling.git_update import _git, repository_blocker

    store = Store(state_dir(root))
    state = priority.read(store)
    if state['phase'] not in {'checkout', 'blocked'}:
        return
    deadline = time.monotonic() + 30
    if reason := repository_blocker(root, deadline):
        raise MigrationError(reason)
    head = _git(root, ['rev-parse', 'HEAD'], deadline)
    if head != state['target']:
        if (retry is None or state.get('version_verified_head') != state['target'] or
                _git(root, ['merge-base', state['target'], head], deadline) != state['target']):
            raise MigrationError('Checkout HEAD differs from the intended Git target')
        failure = status(root, runtime)['failed']
        if failure is None or failure['patch'] != retry:
            raise MigrationError('retry requires the first failed or interrupted patch ID')
    priority.write(store, dict(state, target=head, version_verified=True, version_verified_head=head))


if __name__ == '__main__':
    raise SystemExit(main())
