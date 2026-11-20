"""Publish and receive file updates for stable model builds."""
from __future__ import annotations

import json
import os
from uuid import uuid4

from .core import _copy, atomic_json, digest, fsync_dir, real_tree, safe_path
from src.model_provenance import PROPERTY_FILE, validate_build
from .model_metadata import INVENTORY

INDEX = 'build.json'
ALIAS = '.legacy-build.json'
PENDING = '.build-pending.json'


def load(path, default=None):
    try:
        return json.loads(real_tree(path).read_text())
    except FileNotFoundError:
        return default


def aggregate(folder, manifest, build):
    return load(folder / INDEX) or {
        'host': manifest.get('host') or 'unknown',
        'model': manifest.get('experiment') or manifest['task'],
        'build': build,
        'files': {},
        'runs': {},
        'created_at': '',
        'status': 'success',
    }


def legacy_target(root, manifest):
    name = load(root / ALIAS)
    if not name:
        stamp = manifest.get('created_at', '')
        name = f'build{stamp[:8]}.{stamp[9:15]}.legacy' if stamp else 'build-legacy'
        atomic_json(root / ALIAS, name)
    return real_tree(root / safe_path(name))


def adopt_history(target, manifest):
    """Keep the provisional name pinned until a publisher explicitly claims it."""
    if not manifest.get('adopt_legacy'):
        return
    alias = target.parent / ALIAS
    old = load(alias)
    build = manifest['model_build']
    if not old or old == target.name:
        atomic_json(alias, target.name)
        return
    previous = real_tree(target.parent / safe_path(old))
    if not previous.exists():
        atomic_json(alias, target.name)
        return
    historical = load(previous / INDEX)
    if not historical or not historical['build'].get('estimated'):
        return
    current = load(target / INDEX)
    if current is None and not target.exists():
        previous.rename(target)
        fsync_dir(target.parent)
        current = historical
    else:
        if current is None:
            raise ValueError(f'Formal build has no committed index: {target}')
        for name, file in historical['files'].items():
            if name not in current['files']:
                source = real_tree(previous / safe_path(name))
                if source.is_file():
                    replace_file(source, real_tree(target / safe_path(name)))
                    current['files'][name] = file
        current['runs'] = historical['runs'] | current['runs']
        atomic_json(target / INDEX, current)
        # Keep the provisional original recoverable outside the catalog.
        previous.rename(target.parent / f'.merged-{previous.name}')
        fsync_dir(target.parent)
    current['build'] = build | {'legacy_attribution': 'source_and_model_name'}
    atomic_json(target / INDEX, current)
    atomic_json(alias, target.name)


def replace_file(source, destination):
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError(f'Conflicting model artifact path: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{uuid4().hex}.tmp')
    try:
        _copy(source, temporary)
        os.replace(temporary, destination)
        fsync_dir(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def merge_legacy_log(destination, source):
    """Merge old headed chapters when importing legacy fragments once."""
    import re

    def chapters(text):
        return re.split(r'(?m)(?=^## )', text)

    old = destination.read_text() if destination.is_file() else ''
    known = {part.splitlines()[0] for part in chapters(old) if part.startswith('## ')}
    additions = []
    for part in chapters(source.read_text()):
        if part.startswith('## '):
            key = part.splitlines()[0]
            if key not in known:
                additions.append(part)
                known.add(key)
        elif part.strip() and part not in old:
            additions.append(part)
    if additions:
        temporary = destination.with_name(f'.{destination.name}.{uuid4().hex}.tmp')
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                stream.write(old)
                for part in additions:
                    stream.write('\n' + part)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
            fsync_dir(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)


def apply_update(target, staging, manifest, mapped, experiments, *, legacy=False):
    """Apply a validated package with per-file ordering and a restart marker."""
    identity = ({'name': target.name, 'estimated': True, 'legacy_attribution': 'source_and_model_name'}
                if legacy else validate_build(manifest['model_build']))
    index = aggregate(target, manifest, identity)
    existing = index['build']
    if not legacy:
        if existing.get('fingerprint') not in (None, identity['fingerprint']):
            raise ValueError(f'Model build name already belongs to another fingerprint: {target}')
        if existing.get('legacy_attribution'):
            identity = identity | {'legacy_attribution': existing['legacy_attribution']}
        index['build'] = identity
    if (index['host'], index['model']) != (manifest.get('host') or 'unknown',
                                          manifest.get('experiment') or manifest['task']):
        raise ValueError('Model build belongs to a different source')
    target.mkdir(parents=True, exist_ok=True)
    pending = target / PENDING
    if load(pending, manifest['id']) != manifest['id']:
        raise ValueError('Finish the pending model build update first')
    atomic_json(pending, manifest['id'])
    stamp = manifest.get('created_at') or manifest['batch']
    sequence = manifest.get('build_sequence', 0)
    for item in manifest['files']:
        name = item['path']
        relative = mapped[name].as_posix()
        source = real_tree(staging / relative)
        if (not source.is_file() or relative == INVENTORY or relative.startswith('snapshot/')
                or any(part.startswith('.') for part in safe_path(relative).parts)):
            continue
        fragment = legacy and relative.startswith('run-log/') and relative.split('/')[-1] in ('train.md', 'test.md')
        if fragment:
            relative = 'logs/' + relative.split('/')[-1]
        elif legacy and relative.split('/', 1)[0] not in ('configs', 'logs', 'results', 'html'):
            kind = experiments.get(name)
            relative = f'results/{kind}/{relative}' if kind else 'results/' + relative
        destination = real_tree(target / safe_path(relative))
        previous = index['files'].get(relative, {})
        if previous and (sequence, stamp) < (previous['sequence'], previous['stamp']) and not (
                fragment and previous['sequence'] == 0):
            continue
        if (sequence and previous.get('sequence') == sequence and
                previous.get('package') != manifest['id']):
            raise ValueError('Model update sequence reused by a different package')
        if fragment:
            merge_legacy_log(destination, source)
            if not destination.exists():
                continue
        elif not destination.is_file() or digest(destination) != item['sha256']:
            replace_file(source, destination)
        index['files'][relative] = {
            'path': relative, 'sha256': digest(destination) if fragment else item['sha256'],
            'size': destination.stat().st_size, 'kind': experiments.get(name, 'reports'),
            'stamp': max(stamp, previous.get('stamp', '')) if fragment else stamp,
            'sequence': sequence, 'package': manifest['id'],
        }
    status = manifest.get('status', 'success')
    index['runs'][manifest['run_id']] = {'stamp': stamp, 'status': status, 'package': manifest['id']}
    if stamp >= index['created_at']:
        index.update(created_at=stamp, status=status)
    atomic_json(target / INDEX, index)
    pending.unlink()
    fsync_dir(target)


def archived_builds(config):
    for path in sorted((config.results / 'external/models').glob('*/*/build*/build.json')):
        try:
            index = load(path)
            if not index or path.parent.name != index['build']['name']:
                continue
            if (path.parent / PENDING).exists():
                continue
            yield path.parent, index
        except (OSError, ValueError, TypeError, KeyError):
            continue


def validate_model_members(manifest):
    for file in manifest['files']:
        relative = safe_path(file['path'])
        if len(relative.parts) < 2 or relative.parts[0] not in ('configs', 'logs', 'results', 'html'):
            raise ValueError('Unsupported model build member')
        if any(part.startswith('.') for part in relative.parts):
            raise ValueError('Hidden model build member')
    if manifest.get('adopt_legacy') and not any(i['path'] == PROPERTY_FILE for i in manifest['files']):
        raise ValueError('Historical adoption requires model properties')
