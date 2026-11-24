"""Deliver finished production files into the receiving project's native paths."""
from __future__ import annotations

import json
import os
import re
import zipfile
from concurrent.futures import Future
from contextvars import ContextVar
from functools import wraps
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from .core import (ArtifactSpec, Config, _copy, atomic_json, digest, fsync_dir, locked,
                   publish_artifacts, real_tree, retry, safe_path, sandboxed)

CATEGORY = 'production'
PAYLOAD = 'production'
WEIGHTS = {'state_dict.pt', 'boost_dict.pt', 'boost_head.pt'}
_SAVES = ContextVar('production_artifact_saves', default=None)


def relative_path(name: str):
    """Allow only the concrete weight, prediction and portfolio file layouts."""
    path = safe_path(name)
    parts = path.parts
    if any(part.startswith('.') for part in parts):
        raise ValueError(f'Invalid production member: {name}')
    if len(parts) >= 5 and parts[0] == 'models' and parts[1] in ('nn', 'boost') and '@' in parts[2]:
        if (len(parts) == 8 and parts[3] == 'archive' and parts[4].isdigit()
                and re.fullmatch(r'\d{8}', parts[5]) and path.name in WEIGHTS):
            return path
        if (len(parts) == 5 and parts[3] == 'configs'
                and (path.name in {'model.yaml', 'schedule.yaml'} or re.fullmatch(r'algo\.[\w.-]+\.yaml', path.name))):
            return path
    if parts[:2] == ('data', 'Export'):
        if len(parts) == 6 and parts[2] == 'model_prediction':
            match = re.fullmatch(re.escape(parts[3]) + r'\.(\d{4})\d{4}\.feather', path.name)
            if match and parts[4] == match[1]:
                return path
        if len(parts) == 5 and parts[2] in ('trading_portfolio', 'factor_model_port'):
            if re.fullmatch(re.escape(parts[3]) + r'\.\d{8}\.feather', path.name):
                return path
        if len(parts) == 5 and parts[2] == 'factor_model_account' and path.suffix == '.tar':
            return path
    if parts[:2] == ('results', 'trade'):
        if len(parts) == 4 and parts[2] == 'trading_ports' and re.fullmatch(r'trading_ports\.\d{8}\.csv', path.name):
            return path
        if len(parts) == 5 and parts[2] in ('tracking', 'backtest'):
            if path.name in ('account.tar', f'{parts[3]}_analytic_data.xlsx', f'{parts[3]}_analytic_plot.pdf'):
                return path
        if len(parts) == 6 and parts[2] == 'backtest' and parts[4] == 'portfolio':
            if re.fullmatch(re.escape(parts[3]) + r'\.\d{8}\.feather', path.name):
                return path
    raise ValueError(f'Invalid production member: {name}')


def members(manifest):
    """Validate production metadata before extraction or project writes."""
    if (manifest['schema_version'] != 2 or 'model_build' in manifest
            or any(role['kind'] != 'reports' for role in manifest['artifacts'])):
        raise ValueError('Invalid production package')
    targets = {}
    for item in manifest['files']:
        path = safe_path(item['path'])
        if (path.parts[0] != PAYLOAD or type(item.get('mtime_ns')) is not int
                or not 0 < item['mtime_ns'] < 2**63):
            raise ValueError('Invalid production file metadata')
        targets[item['path']] = relative_path(Path(*path.parts[1:]).as_posix())
    return targets


def finished_report(path):
    """Excel and PDF writers may be visible to reconciliation before closing."""
    if path.suffix == '.xlsx':
        return zipfile.is_zipfile(path)
    if path.suffix == '.pdf':
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 1024))
            return stream.read().rstrip().endswith(b'%%EOF')
    return True


def install(config, manifest, staging, mapped):
    """Install verified bytes with per-file atomic replacement and version checks."""
    root = config.production
    updates = []
    retained = 0
    targets = members(manifest)
    for item in manifest['files']:
        name = item['path']
        target = real_tree(root / targets[name])
        if target.is_relative_to(config.shared_root):
            raise ValueError(f'Production destination is inside the share: {target}')
        stamp = item['mtime_ns']
        previous = target.stat() if target.exists() else None
        if previous is not None:
            if not target.is_file():
                raise ValueError(f'Production destination is not a regular file: {target}')
            if previous.st_mtime_ns >= stamp:
                if previous.st_mtime_ns == stamp and digest(target) != item['sha256']:
                    raise ValueError(f'Conflicting production versions at {target}')
                retained += 1
                continue
        updates.append((staging / mapped[name], target, stamp, previous))

    for source, target, stamp, previous in updates:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev == target.parent.stat().st_dev:
            _install_file(source, target, stamp, previous)
        else:
            with TemporaryDirectory(dir=target.parent, prefix='.production-receive-') as tmp:
                temporary = Path(tmp) / target.name
                _copy(source, temporary)
                _install_file(temporary, target, stamp, previous)
    return {'target': str(root), 'installed': len(updates), 'retained': retained}


def _install_file(source, target, stamp, previous):
    def signature(info):
        return (info.st_ino, info.st_mtime_ns, info.st_size) if info is not None else None

    os.utime(source, ns=(stamp, stamp))
    with source.open('rb') as stream:
        os.fsync(stream.fileno())
    target = real_tree(target)
    current = target.stat() if target.exists() else None
    if signature(current) != signature(previous):
        raise ValueError(f'Production destination changed while receiving: {target}')
    os.replace(source, target)
    fsync_dir(target.parent)


def publisher(config=None):
    if sandboxed():
        return None
    config = config or Config.load()
    if config.enabled and config.role == 'publisher':
        from src.proj import PATH
        if PATH.main == PATH.production:
            return config, PATH.production
    return None


def publish(paths, config, root):
    """Send only new file versions, in generation order, using the existing outbox."""
    if sandboxed() or not config.enabled or config.role != 'publisher':
        return []
    root = real_tree(root)
    ledger = config.state / 'production_published.json'
    output = []
    with locked(config, 'production-publish'):
        previous = json.loads(ledger.read_text()) if ledger.exists() else {}
        candidates = {}
        for source in dict.fromkeys(paths):
            name = relative_path(Path(source).absolute().relative_to(root).as_posix()).as_posix()
            try:
                path = real_tree(source)
                stat = path.stat()
                if not finished_report(path):
                    continue
            except FileNotFoundError:
                continue
            version = [stat.st_mtime_ns, stat.st_size]
            if previous.get(name) != version:
                candidates[name] = version

        def send(batch):
            if not batch:
                return
            run = sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()
            spec = ArtifactSpec(root, category=CATEGORY, members=tuple(batch),
                                experiment=config.source_host, destination=PAYLOAD)
            result = publish_artifacts('production', run, [spec], config=config,
                                       production_versions=batch)
            output.extend(result)
            if result and all(record['status'] == 'published' for record in result):
                previous.update(batch)
                atomic_json(ledger, previous)

        batch = {}
        size = 0
        for name, version in sorted(candidates.items(), key=lambda item: (item[1][0], item[0])):
            if batch and (len(batch) >= min(256, config.max_files - 1)
                          or size + version[1] >= min(256 * 1024**2, config.max_bytes)):
                send(batch)
                batch, size = {}, 0
            batch[name] = version
            size += version[1]
        send(batch)
    return output


def _publish_safe(saves, active):
    from .research import delivery_log

    files = []
    for path, result in saves:
        try:
            completed = result.result() if isinstance(result, Future) else result
            if completed:
                files.append(path)
        except Exception as exc:
            delivery_log(f'production save failed for {path}: {exc}', failed=True)
    if not files:
        return
    try:
        config, root = active
        outcomes = publish(files, config, root)
        for item in outcomes:
            if item['status'] != 'published':
                delivery_log(f'production publication {item["status"]}: {item.get("error") or item.get("reason")}',
                             failed=True)
        if outcomes and all(item['status'] == 'published' for item in outcomes):
            delivery_log(f'published {len(outcomes)} production package(s)')
    except Exception as exc:
        delivery_log(f'production publication failed: {exc}', failed=True)


def _publisher_safe():
    try:
        return publisher()
    except Exception as exc:
        from .research import delivery_log
        delivery_log(f'production publication configuration failed: {exc}', failed=True)
        return None


def batched(function):
    """Group generated versions until this save operation exits."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _SAVES.get() is not None:
            return function(*args, **kwargs)
        active = _publisher_safe()
        if active is None:
            return function(*args, **kwargs)
        queue = []
        token = _SAVES.set((active, queue))
        try:
            return function(*args, **kwargs)
        finally:
            _SAVES.reset(token)
            _publish_safe(queue, active)
    return wrapped


def saved(path, result=True):
    """Record a known successful save or a future for a known production output."""
    if result is False:
        return
    pending = _SAVES.get()
    active = pending[0] if pending is not None else _publisher_safe()
    if active is None:
        return
    try:
        relative_path(Path(path).absolute().relative_to(active[1]).as_posix())
    except ValueError:
        return
    if pending is not None:
        pending[1].append((path, result))
    else:
        _publish_safe([(path, result)], active)


def model_files(base):
    """Select completed inference weights plus the config required to load them."""
    required = 'state_dict.pt' if base.parent.name == 'nn' else 'boost_dict.pt'
    weights = []
    for main in base.glob(f'archive/*/*/*/{required}'):
        if main.is_file() and main.stat().st_size:
            weights.append(main)
            head = main.with_name('boost_head.pt')
            if required == 'state_dict.pt' and head.is_file() and head.stat().st_size:
                weights.append(head)
    if not weights:
        return []
    configs = base / 'configs'
    return weights + [path for path in (configs / 'model.yaml', configs / 'schedule.yaml',
                                       *configs.glob('algo.*.yaml')) if path.is_file()]


def publish_approved_model_safe(base):
    """Called after the lifecycle has committed its production approval."""
    active = _publisher_safe()
    if active is None:
        return
    from .research import delivery_log

    try:
        _publish_safe([(path, True) for path in model_files(base)], active)
    except Exception as exc:
        delivery_log(f'production model inventory failed for {base.name}: {exc}', failed=True)


def sync_current(config=None):
    """Reconcile existing production files and saves missed by a prior publish."""
    active = publisher(config)
    if active is None:
        return []
    config, root = active
    retries = retry(config, category=CATEGORY)
    from src.proj import PATH
    from src.res.model.util.core import ModelPath
    from src.res.model.util.lifecycle import ModelLifecycle
    from .research import delivery_log

    paths = []
    for registration in ModelLifecycle.registrations():
        if registration['stage'] != 'production':
            continue
        try:
            model = ModelLifecycle.approved_path(ModelPath(registration['name']), registered=True)
            paths.extend(model_files(model.base))
        except (OSError, ValueError) as exc:
            delivery_log(f'production model inventory skipped for {registration["name"]}: {exc}', failed=True)
    for directory in (PATH.pred, PATH.trade_port, PATH.fmp_port, PATH.fmp_acc, PATH.rslt_trade):
        for path in directory.rglob('*'):
            if not path.is_file():
                continue
            try:
                relative_path(path.relative_to(root).as_posix())
            except ValueError:
                continue
            paths.append(path)
    return retries + publish(paths, config, root)


def sync_current_safe():
    from .research import delivery_log

    try:
        outcomes = sync_current()
        failed = [item for item in outcomes if item['status'] != 'published']
        if outcomes:
            delivery_log(f'production reconciliation published {len(outcomes) - len(failed)} package(s), '
                         f'{len(failed)} failed', failed=bool(failed))
        for item in failed:
            delivery_log(str(item.get('error') or item.get('reason')), failed=True)
    except Exception as exc:
        delivery_log(f'production reconciliation failed: {exc}', failed=True)
