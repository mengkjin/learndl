"""Immutable, checked filesystem delivery; importing this module has no project side effects."""
from __future__ import annotations
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import stat
import tempfile
import time
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from src.file_ops import atomic_json, fsync_dir, real_tree

__all__ = [
    'MAX_BYTES',
    'MAX_FILES',
    'CATEGORIES',
    'ROOT',
    'DEFAULT_CONFIG',
    'atomic_json',
    'fsync_dir',
    'digest',
    'safe_path',
    'slug',
    'real_tree',
    'Config',
    'sandboxed',
    'locked',
    'ArtifactSpec',
    'valid_category',
    'component',
    'route',
    'checked_entries',
    'publish_artifacts',
    'send_prepared',
    'retry',
    'category_root',
    'archive_target',
    'archived_batches',
    'experiment_members',
    'received_member',
    'receive',
]

MAX_BYTES = 4 * 1024**3
MAX_FILES = 50000
CATEGORIES = {'model_training', 'model_compare', 'factor_sequence', 'rl', 'custom', 'storage', 'data_quality'}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / '.local_resources/artifact_exchange/config.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(value):
    p = PurePosixPath(value)
    if (not value or p.is_absolute() or '..' in p.parts or '.' in value.split('/')
            or any(c in value for c in '\\:\x00') or any(ord(c) < 32 for c in value)
            or p.as_posix() != value):
        raise ValueError(f'Unsafe relative path: {value!r}')
    return p


def slug(value):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(value)).strip('._-')[:64] or 'unnamed'


@dataclass
class Config:
    enabled: bool = False
    role: str = 'receiver'
    shared_root: Path = Path.home() / 'workspace/Syncthing'
    state: Path = ROOT / '.local_resources/artifact_exchange'
    results: Path = ROOT / 'results'
    max_bytes: int = MAX_BYTES
    max_files: int = MAX_FILES
    source_host: str = field(default_factory=socket.gethostname)

    def __post_init__(self):
        for name in ('shared_root', 'state', 'results'):
            p = Path(os.path.expandvars(os.path.expanduser(str(getattr(self, name)))))
            if not p.is_absolute():
                raise ValueError(f'{name} must be absolute')
            setattr(self, name, real_tree(p))
        if self.role not in ('publisher', 'receiver') or self.max_bytes <= 0 or self.max_files <= 0:
            raise ValueError('Invalid role or limits')
        for p in (self.state, self.results):
            if p.is_relative_to(self.shared_root) or self.shared_root.is_relative_to(p):
                raise ValueError('Shared and private paths must not overlap')

    @classmethod
    def load(cls, path=None):
        p = Path(path or os.environ.get('LEARNDL_ARTIFACT_CONFIG', DEFAULT_CONFIG))
        values = json.loads(p.read_text()) if p.exists() else {}
        config = cls(**values)
        identity = config.state / 'source_host.json'
        if 'source_host' not in values and identity.exists():
            config.source_host = json.loads(identity.read_text())['source_host']
        return config

    @property
    def inbox(self):
        return self.shared_root / 'artifact_exchange'


def sandboxed():
    return bool(os.environ.get('LEARNDL_AUTOFIX_ID')) or Path('/run/learndl-autofix.json').exists()


@contextlib.contextmanager
def locked(config, name):
    real_tree(config.state).mkdir(parents=True, exist_ok=True)
    with (config.state / (name + '.lock')).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@dataclass(frozen=True)
class ArtifactSpec:
    source: Path
    kind: str = 'reports'
    category: str = 'custom'
    name: str = ''
    members: tuple[str, ...] = ()
    experiment: str = ''
    destination: str = ''
    run_id: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'source', Path(self.source).absolute())
        if self.kind not in ('reports', 'tensorboard', 'optuna') or not valid_category(self.category):
            raise ValueError('Unknown artifact route')
        if self.destination:
            safe_path(self.destination)


def valid_category(value):
    return isinstance(value, str) and bool(re.fullmatch('[a-z][a-z0-9_]{0,47}', value)) and value not in {
        'reports', 'tensorboard', 'optuna', 'model_experiments', 'source'}


def component(value):
    """Readable filesystem identity without collapsing distinct original names."""
    value = str(value)
    clean = re.sub(r'[^a-zA-Z0-9_.@-]+', '_', value).strip('._-')[:96] or 'unnamed'
    return clean if clean == value else clean + '_' + hashlib.sha256(value.encode()).hexdigest()[:10]


def exchange_category(category: str) -> str:
    return 'models' if category in ('model_training', 'model') else category


def archive_category(category: str) -> str:
    return {'model_training': 'models', 'model': 'models', 'rl': 'rl_experiment',
            'factor_sequence_fixes': 'factor_sequence'}.get(category, category)


def route(kind, category):
    if kind not in ('reports', 'tensorboard', 'optuna') or category not in CATEGORIES:
        raise ValueError('Unknown route')
    return Path('reports') / category if kind == 'reports' else Path(kind)


def checked_entries(archive, config):
    entries = archive.infolist()
    if len(entries) > config.max_files or sum(i.file_size for i in entries) > config.max_bytes:
        raise ValueError('Archive exceeds configured limits')
    names = set()
    files = set()
    for item in entries:
        name = item.filename.rstrip('/') if item.is_dir() else item.filename
        safe_path(name)
        key = unicodedata.normalize('NFC', name).casefold()
        mode = item.external_attr >> 16
        if key in names or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
            raise ValueError('Duplicate, conflicting or special ZIP member')
        names.add(key)
        if not item.is_dir():
            files.add(key)
    if any(any(str(p) in files for p in PurePosixPath(n).parents if str(p) != '.') for n in names):
        raise ValueError('File/directory collision')
    return entries


def _copy(source, target):
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'Source changed during snapshot: {source}')


def _snapshot(spec, payload, config):
    source = real_tree(spec.source)
    if not source.exists():
        raise FileNotFoundError(source)
    if spec.kind == 'optuna':
        target = payload / (component(spec.name or source.stem) + '.zip')
        snap = payload.parent / 'snapshot.sqlite3'
        deadline = time.monotonic() + 60
        def progress(*args):
            if time.monotonic() > deadline:
                raise TimeoutError('SQLite backup exceeded 60 seconds')
        with contextlib.closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as db:
            with contextlib.closing(sqlite3.connect(snap)) as dst:
                db.backup(dst, pages=256, progress=progress)
        if snap.stat().st_size > config.max_bytes:
            raise ValueError('Database exceeds limit')
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
            _write_zip_member(z, snap, 'study.sqlite3')
        return
    sources = ([source / safe_path(n) for n in spec.members] if spec.members else
               sorted(source.rglob('*')) if source.is_dir() else [source])
    selected = []
    for p in sources:
        real_tree(p)
        if p.is_dir():
            continue
        if not stat.S_ISREG(p.stat().st_mode):
            raise ValueError('Special source file refused')
        selected.append(p)
    if not selected or len(selected) > config.max_files or sum(p.stat().st_size for p in selected) > config.max_bytes:
        raise ValueError('Empty source or source exceeds limits')
    signature = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) for p in selected}
    work = payload if spec.kind == 'reports' else payload.parent / 'events'
    for p in selected:
        rel = p.relative_to(source) if source.is_dir() else Path(p.name)
        safe_path(rel.as_posix())
        _copy(p, work / rel)
    if any((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) != signature[p] for p in selected):
        raise ValueError('Source changed during batch snapshot')
    if source.is_dir() and not spec.members:
        if set(p for p in source.rglob('*') if not p.is_dir()) != set(selected):
            raise ValueError('Source inventory changed during snapshot')
    if spec.kind == 'tensorboard':
        if not any('tfevents' in p.name for p in selected):
            raise ValueError('No TensorBoard events')
        with zipfile.ZipFile(payload / (component(spec.name or source.name) + '.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(work.rglob('*')):
                if p.is_file():
                    _write_zip_member(z, p, 'tensorboard/' + p.relative_to(work).as_posix())


def _write_zip_member(archive, source, name):
    entry = zipfile.ZipInfo(name)
    entry.compress_type = zipfile.ZIP_DEFLATED
    with source.open('rb') as stream, archive.open(entry, 'w', force_zip64=True) as output:
        shutil.copyfileobj(stream, output, 1024 * 1024)


def publish_artifacts(task, run_id, artifacts, status='success', *, config=None, prepare_only=False, build=None, adopt_legacy=False) -> list[dict[str, Any]]:
    if sandboxed():
        return [{'status': 'skipped', 'reason': 'autofix'}]
    config = config or Config.load()
    if not config.enabled or config.role != 'publisher':
        return [{'status': 'skipped', 'reason': 'publisher disabled'}]
    outcomes = []
    groups = {}
    for spec in artifacts:
        groups.setdefault((spec.category, spec.experiment or str(task), spec.run_id or str(run_id)), []).append(spec)
    with locked(config, 'publish'):
        atomic_json(config.state / 'source_host.json', {'source_host': config.source_host})
        for (category, experiment, group_run), specs in groups.items():
            try:
                outcomes.append(_prepare(task, group_run, list(dict.fromkeys(specs)), status, config, category, experiment, prepare_only, build, adopt_legacy))
            except Exception as exc:
                failure = {'status': 'failed', 'sources': [str(s.source) for s in specs], 'error': str(exc)}
                atomic_json(config.state / 'failures' / (uuid.uuid4().hex + '.json'), failure)
                outcomes.append(failure)
    return outcomes


def _prepare(task, run_id, specs, status, config, category, experiment, prepare_only=False, build=None, adopt_legacy=False) -> dict[str, Any]:
    # Mail callers use private temporary directories. The notification ID
    # identifies the logical message even when it is retried from a new directory.
    declaration = [] if category == 'mail_archive' else [asdict(s) | {'source': str(s.source)} for s in specs]
    identity = [2, config.source_host, task, run_id, category, experiment, declaration]
    if build is not None:
        from src.model_provenance import validate_build

        validate_build(build)
        if category != 'model_training':
            raise ValueError('Model build updates require model_training')
        identity = [2, config.source_host, task, run_id, category, experiment, build['fingerprint']]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    ledger = config.state / 'outbox' / key / 'delivery.json'
    if ledger.exists():
        record = json.loads(ledger.read_text())
        record['outbox_key'] = key
        return record if prepare_only else _send(record, ledger, config)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ledger.parent) as tmp:
        work = Path(tmp)
        payload = work / 'payload'
        payload.mkdir()
        artifacts = []
        source_files = []
        # Isolate scratch databases/events for each spec; merge only checked payloads.
        for index, spec in enumerate(specs):
            part = work / str(index) / 'payload'
            part.mkdir(parents=True)
            _snapshot(spec, part, config)
            destination = spec.destination or ('results' if spec.kind == 'reports' else spec.kind)
            if category == 'model_training' and (build is not None or task == 'model_training') and spec.kind != 'reports' and not spec.destination:
                destination = f'results/{spec.kind}'
            safe_path(destination)
            paths = []
            for source in sorted(part.rglob('*')):
                if not source.is_file():
                    continue
                relative = Path(destination) / source.relative_to(part)
                target = payload / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f'Overlapping artifact destination: {relative}')
                if spec.kind == 'reports':
                    original = spec.source / source.relative_to(part) if spec.source.is_dir() else spec.source
                    source_files.append({'source': str(original), 'sha256': digest(source)})
                source.rename(target)
                paths.append(relative.as_posix())
            artifacts.append({'kind': spec.kind, 'name': spec.name or spec.source.name,
                              'destination': destination, 'paths': paths})
        package_id = uuid.uuid4().hex
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        batch = stamp + '__' + package_id
        files = [{'path': p.relative_to(payload).as_posix(), 'size': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(payload.rglob('*')) if p.is_file()]
        if build is not None:
            state_path = real_tree(config.state / 'model_updates' / component(config.source_host) /
                                   component(experiment) / (build['fingerprint'] + '.json'))
            state = json.loads(state_path.read_text()) if state_path.exists() else {'sequence': 0, 'files': {}}
            # Persist sequence reservations before preparing the immutable outbox.
            # A crash can resend unchanged files but can never reuse a sequence.
            files = [item for item in files
                     if adopt_legacy or state['files'].get(item['path']) != item['sha256']]
            if not files:
                return {'status': 'skipped', 'reason': 'model files unchanged'}
            members = {item['path'] for item in files}
            artifacts = [artifact | {'paths': [p for p in artifact['paths'] if p in members]}
                         for artifact in artifacts if any(p in members for p in artifact['paths'])]
            state['sequence'] += 1
            atomic_json(state_path, state)
        manifest = {'schema_version': 2, 'id': package_id, 'task': str(task),
                    'run_id': str(run_id), 'batch': batch, 'created_at': stamp, 'category': category,
                    'experiment': experiment, 'experiment_dir': component(experiment),
                    'status': status, 'host': config.source_host, 'host_dir': component(config.source_host),
                    'artifacts': artifacts, 'files': files}
        if build is not None:
            from .model_builds import validate_model_members

            manifest.update(model_build=build, build_sequence=state['sequence'], adopt_legacy=adopt_legacy)
            validate_model_members(manifest)
        filename = batch + '.transfer.zip'
        archive = ledger.parent / 'package.zip'
        with zipfile.ZipFile(work / 'package.zip', 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
            z.writestr('manifest.json', json.dumps(manifest))
            for item in files:
                z.write(payload / item['path'], 'payload/' + item['path'])
        with zipfile.ZipFile(work / 'package.zip') as z:
            checked_entries(z, config)
        os.replace(work / 'package.zip', archive)
        with archive.open('rb') as f:
            os.fsync(f.fileno())
        record = {'id': package_id, 'status': 'prepared', 'filename': filename, 'outbox_key': key,
                  'route': (Path(exchange_category(category)) / component(experiment)).as_posix(), 'sha256': digest(archive),
                  'sources': [str(s.source) for s in specs], 'source_files': source_files, 'manifest': manifest}
        atomic_json(ledger, record)
        if build is not None:
            state['files'].update((item['path'], item['sha256']) for item in files)
            atomic_json(state_path, state)
    return record if prepare_only else _send(record, ledger, config)


def send_prepared(records, config):
    """Publish only this task's frozen batches, updating references held by mail callbacks."""
    if sandboxed() or not config.enabled or config.role != 'publisher':
        for record in records:
            if record.get('status') == 'prepared':
                record.update(status='skipped', reason='publisher disabled or autofix')
        return records
    with locked(config, 'publish'):
        for record in records:
            if record.get('status') != 'prepared':
                continue
            try:
                key = record['outbox_key']
                if not re.fullmatch('[0-9a-f]{64}', key):
                    raise ValueError('Invalid outbox identity')
                ledger = config.state / 'outbox' / key / 'delivery.json'
                frozen = json.loads(ledger.read_text())
                if frozen['id'] != record['id']:
                    raise ValueError('Outbox identity changed')
                record.update(_send(frozen, ledger, config))
            except Exception as exc:
                record.update(status='failed', error=str(exc))
    return records


def _send(record, ledger, config) -> dict[str, Any]:
    if record['status'] == 'published':
        (ledger.parent / 'package.zip').unlink(missing_ok=True)
        return record
    if not (config.shared_root / '.stfolder').exists():
        raise ValueError('Syncthing folder marker missing; configure shared root first')
    archive = ledger.parent / 'package.zip'
    if digest(archive) != record['sha256']:
        raise ValueError('Spool checksum mismatch')
    target = real_tree(config.inbox / record['route'] / record['filename'])
    target.parent.mkdir(parents=True, exist_ok=True)
    # Outside the shared root, but on its filesystem for atomic publication.
    with tempfile.TemporaryDirectory(dir=config.shared_root.parent, prefix='.artifact-publish-') as tmp:
        staging = Path(tmp) / record['filename']
        _copy(archive, staging)
        if target.exists() and digest(target) != record['sha256']:
            raise ValueError('Publication conflict')
        for attempt in range(3):
            try:
                real_tree(target.parent).mkdir(parents=True, exist_ok=True)
                os.replace(staging, target)
                break
            except FileNotFoundError:
                # Synced empty-directory deletion may race with publication.
                if attempt == 2 or not staging.exists():
                    raise
        fsync_dir(target.parent)
    record['status'] = 'published'
    atomic_json(ledger, record)
    archive.unlink(missing_ok=True)
    return record


def retry(config):
    if sandboxed() or not config.enabled or config.role != 'publisher':
        return []
    output = []
    with locked(config, 'publish'):
        for p in sorted((config.state / 'outbox').glob('*/delivery.json')):
            record = json.loads(p.read_text())
            if record['status'] != 'published':
                try:
                    output.append(_send(record, p, config))
                except Exception as exc:
                    output.append({'id': record['id'], 'status': 'failed', 'error': str(exc)})
    return output


def category_root(results, category):
    """Honor only the exact legacy report alias created by our migration tool."""
    root = real_tree(results) / category
    legacy = results / 'reports' / category / 'legacy' / category
    if root.is_symlink() and root.resolve() == legacy:
        return real_tree(legacy)
    return real_tree(root)


def _validate_identity(manifest):
    if not re.fullmatch('[0-9a-f]{32}', manifest['id']):
        raise ValueError('Invalid package ID')
    if manifest['schema_version'] == 2:
        if (not isinstance(manifest['host'], str) or not manifest['host'] or
                not isinstance(manifest['experiment'], str) or not manifest['experiment'] or
                not valid_category(manifest['category']) or
                component(manifest['experiment']) != manifest['experiment_dir'] or
                component(manifest['host']) != manifest['host_dir'] or
                not re.fullmatch(r'[0-9]{8}T[0-9]{12}Z', manifest['created_at']) or
                manifest['batch'] != manifest['created_at'] + '__' + manifest['id']):
            raise ValueError('Invalid experiment identity')
        return
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported schema')
    route(manifest['kind'], manifest['category'])
    if slug(manifest['task']) != manifest['task'] or manifest['batch'] != slug(manifest['run_id']) + '__' + manifest['id']:
        raise ValueError('Invalid task/batch')


def _target(manifest, config):
    """Historical target, also used by immutable pre-0.7.4 version patches."""
    _validate_identity(manifest)
    if manifest['schema_version'] == 2:
        return real_tree(category_root(config.results, manifest['category']) /
                         (manifest['host_dir'] + '__' + manifest['experiment_dir']) / manifest['batch'])
    base = config.results / ('reports' if manifest['kind'] == 'reports' else 'model_experiments')
    return real_tree(base / (manifest['category'] if manifest['kind'] == 'reports' else manifest['kind']) / manifest['task'] / manifest['batch'])


def archive_target(manifest, config):
    _validate_identity(manifest)
    if manifest['schema_version'] == 2:
        host, experiment = manifest['host_dir'], manifest['experiment_dir']
    else:
        host = component(manifest.get('host') or 'unknown')
        experiment = component(manifest.get('experiment') or manifest['task'])
    root = real_tree(config.results / 'external' / archive_category(manifest['category']) / host / experiment)
    if 'model_build' in manifest:
        from src.model_provenance import validate_build

        if manifest['category'] != 'model_training' or type(manifest.get('build_sequence')) is not int or manifest['build_sequence'] < 1:
            raise ValueError('Invalid model build update')
        return real_tree(root / validate_build(manifest['model_build'])['name'])
    return real_tree(root / manifest['batch'])


def archived_batches(config, category=None):
    """Discover canonical batches only, excluding staged or relocated copies."""
    external = real_tree(config.results / 'external')
    pattern = (archive_category(category) if category else '*') + '/*/*/*/source.json'
    for source in sorted(external.glob(pattern), reverse=True):
        try:
            manifest = json.loads(real_tree(source).read_text())
            if category is not None and manifest['category'] != category:
                continue
            if archive_target(manifest, config) == source.parent:
                yield source.parent, manifest
        except (OSError, ValueError, TypeError, KeyError):
            continue


def experiment_members(manifest):
    """Validate v2 role declarations against the complete payload inventory."""
    if manifest['schema_version'] == 1:
        return {i['path']: manifest['kind'] for i in manifest['files'] if manifest['kind'] != 'reports'}
    declared = set()
    experiments = {}
    for artifact in manifest['artifacts']:
        if artifact['kind'] not in ('reports', 'tensorboard', 'optuna'):
            raise ValueError('Unknown artifact kind')
        destination = safe_path(artifact['destination'])
        if not artifact['paths']:
            raise ValueError('Empty artifact')
        for name in artifact['paths']:
            path = safe_path(name)
            if name in declared or destination not in path.parents:
                raise ValueError('Overlapping or invalid artifact paths')
            declared.add(name)
            if artifact['kind'] != 'reports':
                if path.suffix != '.zip':
                    raise ValueError('Experiment payload must be a ZIP')
                experiments[name] = artifact['kind']
    if declared != {i['path'] for i in manifest['files']}:
        raise ValueError('Artifact inventory mismatch')
    return experiments


def received_member(manifest, name, experiments, *, folder=None):
    """Map historical model members without changing the recorded inventory."""
    original = safe_path(name)
    kind = experiments.get(name)
    if manifest.get('schema_version') == 2 and manifest.get('category') == 'model_training':
        canonical = None
        if original.parts[:2] == ('results', 'factor_residual'):
            canonical = Path('snapshot', *original.parts[1:])
        elif kind and original.parts[0] == kind:
            canonical = Path('results') / original
        if canonical is not None and (folder is None or real_tree(folder / canonical).exists()):
            return canonical
    return original


def _verify_target(target, manifest):
    if json.loads((target / 'source.json').read_text()) != manifest:
        raise ValueError('Existing target has different provenance')
    experiments = experiment_members(manifest)
    for item in manifest['files']:
        p = real_tree(target / received_member(manifest, item['path'], experiments, folder=target))
        if p.stat().st_size != item['size'] or digest(p) != item['sha256']:
            raise ValueError('Existing target content differs; package retained')


def _receive(archive, config):
    real_tree(archive)
    archive_hash = digest(archive)
    with zipfile.ZipFile(archive) as z:
        entries = checked_entries(z, config)
        if z.getinfo('manifest.json').file_size > 16 * 1024**2:
            raise ValueError('Oversized manifest')
        manifest = json.loads(z.read('manifest.json'))
        if manifest['schema_version'] not in (1, 2):
            raise ValueError('Unsupported schema')
        target = archive_target(manifest, config)
        build_update = 'model_build' in manifest
        legacy_update = (manifest['category'] == 'model_training' and not build_update
                         and (config.state / 'model_builds_ready.json').is_file())
        if build_update:
            from .model_builds import validate_model_members

            validate_model_members(manifest)
        experiments = experiment_members(manifest)
        expected_route = (Path(manifest['category']) / manifest['experiment_dir'] if manifest['schema_version'] == 2
                          else route(manifest['kind'], manifest['category']))
        routes = {expected_route}
        if manifest['schema_version'] == 2:
            routes.add(Path(exchange_category(manifest['category'])) / manifest['experiment_dir'])
            if manifest['category'] == 'model_training':
                routes.add(Path('model') / manifest['experiment_dir'])
        if archive.parent.relative_to(config.inbox) not in routes:
            raise ValueError('Route mismatch')
        listed = {}
        for item in manifest['files']:
            safe_path(item['path'])
            if item['path'].casefold() == 'source.json' or item['path'] in listed:
                raise ValueError('Reserved or duplicate manifest member')
            listed[item['path']] = item
        mapped = {name: received_member(manifest, name, experiments) for name in listed}
        paths = {str(p).casefold() for p in mapped.values()}
        if len(paths) != len(mapped) or any(str(parent).casefold() in paths
                                           for p in mapped.values() for parent in p.parents):
            raise ValueError('Overlapping received artifact paths')
        if not listed or set(i.filename for i in entries) != {'manifest.json', *('payload/' + n for n in listed)}:
            raise ValueError('Manifest/payload mismatch')
        receipt = config.state / 'received' / (manifest['id'] + '.json')
        previous = json.loads(receipt.read_text()) if receipt.exists() else None
        if previous and previous['sha256'] != archive_hash:
            raise ValueError('Package ID reused with different content')
        if previous and (build_update or legacy_update):
            return _finish_receive(archive, archive_hash, manifest, previous['target'], build_archive=True)
        if previous and manifest.get('host') == config.source_host:
            legacy = config.results / 'custom/legacy' / _target(manifest, config).relative_to(config.results)
            if previous['target'] == str(legacy):
                target = real_tree(legacy)
        if previous and manifest['schema_version'] == 2 and manifest.get('category') in ('model_training', 'mail_archive'):
            return _finish_receive(archive, archive_hash, manifest, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix='.receive-') as tmp:
            staging = Path(tmp) / 'result'
            staging.mkdir()
            for name, item in listed.items():
                info = z.getinfo('payload/' + name)
                if info.file_size != item['size']:
                    raise ValueError('Manifest size mismatch')
                p = staging / mapped[name]
                p.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, p.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                if digest(p) != item['sha256']:
                    raise ValueError('Payload checksum mismatch')
                if name in experiments:
                    with zipfile.ZipFile(p) as inner:
                        checked_entries(inner, config)
                        if inner.testzip() is not None:
                            raise ValueError('Damaged experiment archive')
            if build_update or legacy_update:
                from .model_builds import adopt_history, apply_update, legacy_target

                if legacy_update:
                    target = legacy_target(target.parent, manifest)
                else:
                    adopt_history(target, manifest)
                apply_update(target, staging, manifest, mapped, experiments, legacy=legacy_update)
                atomic_json(receipt, {'id': manifest['id'], 'sha256': archive_hash, 'target': str(target)})
                return _finish_receive(archive, archive_hash, manifest, target, build_archive=True)
            atomic_json(staging / 'source.json', manifest)
            for directory in sorted((p for p in staging.rglob('*') if p.is_dir()), reverse=True):
                fsync_dir(directory)
            fsync_dir(staging)
            if target.exists():
                _verify_target(target, manifest)
            else:
                staging.rename(target)
                fsync_dir(target.parent)
        if manifest['schema_version'] == 2 and manifest['category'] == 'model_training':
            from .model_logs import append_received

            append_received(config, target, manifest)
        atomic_json(receipt, {'id': manifest['id'], 'sha256': archive_hash, 'target': str(target)})
    return _finish_receive(archive, archive_hash, manifest, target)


def _finish_receive(archive, archive_hash, manifest, target, *, build_archive=False):
    if not build_archive and manifest['schema_version'] == 2 and manifest.get('category') == 'model_training':
        fragments = {'run-log/train.md', 'run-log/test.md'}
        members = {item['path'] for item in manifest['files']}
        for name in members & fragments:
            real_tree(target / name).unlink(missing_ok=True)
        folder = target / 'run-log'
        with contextlib.suppress(FileNotFoundError):
            if members & fragments and not any(folder.iterdir()):
                folder.rmdir()
                fsync_dir(target)
        if members and members <= fragments:
            with contextlib.suppress(FileNotFoundError):
                real_tree(target / 'source.json').unlink(missing_ok=True)
                if not any(target.iterdir()):
                    target.rmdir()
                    fsync_dir(target.parent)
    if digest(archive) != archive_hash:
        raise ValueError('Package changed before cleanup')
    archive.unlink()
    fsync_dir(archive.parent)
    return {'status': 'received', 'id': manifest['id'], 'target': str(target)}


def _prune_empty_routes(config):
    """Remove only empty protocol route directories, never the inbox or share root."""
    errors = []

    def remove(directory):
        try:
            real_tree(directory)
            directory.rmdir()  # Atomic emptiness check; concurrent arrivals win.
        except (FileNotFoundError, ValueError):
            pass
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOTDIR):
                errors.append({'status': 'cleanup_failed', 'directory': str(directory), 'error': str(exc)})

    for category in sorted(config.inbox.glob('*')):
        if category.is_symlink() or not category.is_dir():
            continue
        if category.name in ('tensorboard', 'optuna'):
            remove(category)  # Legacy v1 routes have no experiment level.
        elif category.name == 'reports' or valid_category(category.name):
            for experiment in sorted(category.glob('*')):
                if experiment.is_symlink() or not experiment.is_dir():
                    continue
                if (experiment.name in CATEGORIES if category.name == 'reports' else
                        not experiment.name.startswith('.') and component(experiment.name) == experiment.name):
                    remove(experiment)
            remove(category)
    return errors


def receive(config):
    if sandboxed() or not config.enabled or config.role != 'receiver':
        return []
    output = []
    with locked(config, 'receive'):
        if not (config.shared_root / '.stfolder').exists():
            raise ValueError('Syncthing folder marker missing')
        real_tree(config.inbox)
        for p in sorted(config.inbox.rglob('*.transfer.zip')):
            try:
                output.append(_receive(p, config))
            except Exception as exc:
                output.append({'status': 'failed', 'file': str(p), 'error': str(exc)})
        output.extend(_prune_empty_routes(config))
        atomic_json(config.state / 'last_receive.json', output)
    return output
