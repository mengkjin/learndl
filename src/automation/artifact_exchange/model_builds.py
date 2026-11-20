"""Publish and receive file updates for stable model builds."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from uuid import uuid4

from .archive_paths import update_state
from .core import _copy, atomic_json, digest, fsync_dir, real_tree, safe_path
from src.shared.model_provenance import LOG_PHASES, PROPERTY_FILE, validate_build
from .model_metadata import INVENTORY

__all__ = [
    'ALIAS',
    'INDEX',
    'PENDING',
    'LEGACY_LOG_MEMBERS',
    'load',
    'legacy_target',
    'legacy_member',
    'adopt_history',
    'merge_legacy_log',
    'apply_update',
    'archived_builds',
    'validate_model_members',
]

INDEX = 'build.json'
ALIAS = '.legacy-build.json'
PENDING = '.build-pending.json'
LEGACY_LOG_MEMBERS = frozenset(f'run-log/{phase}.md' for phase in LOG_PHASES)


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


def locate_build(config, manifest, target):
    """Resolve the receiver's local name and archive location from committed indexes."""
    host = manifest.get('host') or 'unknown'
    model = manifest.get('experiment') or manifest['task']
    fingerprint = manifest.get('model_build', {}).get('fingerprint')
    matches = [folder for folder, index in archived_builds(config, pending=True)
               if index['host'] == host and index['model'] == model
               and (index['build'].get('fingerprint') == fingerprint if fingerprint else folder.name == target.name)]
    if len(matches) > 1:
        raise ValueError('Multiple local locations for this build')
    return matches[0] if matches else target


def adopt_history(target, manifest, config, monitor, origin=None):
    """Keep the provisional name pinned until a publisher explicitly claims it."""
    if not manifest.get('adopt_legacy'):
        return target
    alias = (origin or target).parent / ALIAS
    journal = alias.with_name('.legacy-adoption.json')
    build = manifest['model_build']
    planned = load(journal)
    if planned:
        if planned['fingerprint'] != build['fingerprint']:
            raise ValueError('Another model build adoption is pending')
        old = planned['old']
        previous, target = real_tree(planned['source']), real_tree(planned['target'])
        current = load(target / INDEX)
    else:
        old = load(alias)
        if not old or old == target.name:
            atomic_json(alias, target.name)
            return target
        previous = locate_build(config, {k: v for k, v in manifest.items() if k != 'model_build'},
                                real_tree(alias.parent / safe_path(old)))
        current = load(target / INDEX)
        if previous.exists() and previous.parent != target.parent:
            if current is not None:
                raise ValueError('Formal and legacy builds occupy different local locations')
            target = previous.with_name(target.name)
            current = load(target / INDEX)
    if current and current['build'].get('fingerprint') not in (None, build['fingerprint']):
        raise ValueError(f'Formal build has a different fingerprint: {target}')
    if previous.exists():
        historical = load(previous / INDEX)
        if not historical or not historical['build'].get('estimated'):
            return target
        if (previous / PENDING).exists() or (target / PENDING).exists():
            raise ValueError('Finish pending updates before adopting model history')
        atomic_json(journal, {'fingerprint': build['fingerprint'], 'old': old,
                              'source': str(previous), 'target': str(target)})
        if current is None:
            previous.rename(target)
            fsync_dir(target.parent)
            current = historical
        else:
            for name, file in historical['files'].items():
                if name not in current['files']:
                    source = real_tree(previous / safe_path(name))
                    if source.is_file():
                        replace_file(source, real_tree(target / safe_path(name)))
                    current['files'][name] = file
            current['runs'] = historical['runs'] | current['runs']
            atomic_json(target / INDEX, current)
            previous.rename(target.parent / f'.merged-{previous.name}')
            fsync_dir(target.parent)
    if current is None:
        raise ValueError(f'Missing legacy build index: {previous}')
    current['build'] = build | {'legacy_attribution': 'source_and_model_name'}
    atomic_json(target / INDEX, current)
    update_state(config, monitor, {previous: target})
    marks = monitor / 'model_preview/model_views.json'
    views = load(marks, {})
    key = sha256(json.dumps([current['host'], current['model']]).encode()).hexdigest()
    if old in views.get(key, {}):
        views[key][target.name] = views[key].pop(old)
        atomic_json(marks, views)
    atomic_json(alias, target.name)
    journal.unlink(missing_ok=True)
    fsync_dir(journal.parent)
    return target


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
                for position, part in enumerate(additions):
                    if old or position:
                        stream.write('\n')
                    stream.write(part)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
            fsync_dir(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)


def legacy_member(relative, kind=None):
    """Return the build-relative destination of an old batch member."""
    if relative in LEGACY_LOG_MEMBERS:
        return 'logs/' + relative.split('/')[-1]
    if relative.split('/', 1)[0] not in ('configs', 'logs', 'results', 'html'):
        return f'results/{kind}/{relative}' if kind else 'results/' + relative
    return relative


def apply_update(target, staging, manifest, mapped, experiments, *, legacy=False, trashed_members=()):
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
    # Protocol v1 has no creation time. Keep it behind dated updates.
    stamp = manifest.get('created_at') or '19700101T000000000000Z'
    sequence = manifest.get('build_sequence', 0)
    for item in manifest['files']:
        name = item['path']
        relative = mapped[name].as_posix()
        source = real_tree(staging / relative)
        trashed = name in trashed_members and not source.exists()
        if ((not trashed and not source.is_file()) or relative == INVENTORY or relative.startswith('snapshot/')
                or any(part.startswith('.') for part in safe_path(relative).parts)):
            continue
        fragment = legacy and relative in LEGACY_LOG_MEMBERS
        if legacy:
            relative = legacy_member(relative, experiments.get(name))
        destination = real_tree(target / safe_path(relative))
        previous = index['files'].get(relative, {})
        if previous and (sequence, stamp) < (previous['sequence'], previous['stamp']) and not (
                fragment and previous['sequence'] == 0):
            continue
        if (sequence and previous.get('sequence') == sequence and
                previous.get('package') != manifest['id']):
            raise ValueError('Model update sequence reused by a different package')
        if trashed:
            # Reserve the indexed path for a future restore without reviving a deleted file.
            if destination.exists() or fragment:
                continue
        elif fragment:
            merge_legacy_log(destination, source)
            if not destination.exists():
                continue
        elif not destination.is_file() or digest(destination) != item['sha256']:
            replace_file(source, destination)
        index['files'][relative] = {
            'path': relative, 'sha256': digest(destination) if fragment else item['sha256'],
            'size': item['size'] if trashed else destination.stat().st_size, 'kind': experiments.get(name, 'reports'),
            'stamp': max(stamp, previous.get('stamp', '')) if fragment else stamp,
            'sequence': sequence, 'package': manifest['id'], 'run': manifest['run_id'],
        }
    status = manifest.get('status', 'success')
    index['runs'][manifest['run_id']] = {'stamp': stamp, 'status': status, 'package': manifest['id']}
    if stamp >= index['created_at']:
        index.update(created_at=stamp, status=status)
    atomic_json(target / INDEX, index)
    pending.unlink()
    fsync_dir(target)


def archived_builds(config, *, pending=False):
    root = config.results / 'external/models'
    paths = (*root.glob('*/*/build*/build.json'), *(root / 'archive').glob('*/*/build*/build.json'))
    for path in sorted(paths):
        try:
            index = load(path)
            if not index or path.parent.name != index['build']['name']:
                continue
            if not pending and (path.parent / PENDING).exists():
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
