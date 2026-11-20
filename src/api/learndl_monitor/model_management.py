"""Operations on local training folders and individually received model builds."""
from __future__ import annotations

from contextlib import ExitStack, suppress
from datetime import datetime, timedelta
from hashlib import sha256
from uuid import uuid4
from datetime import timezone
from pathlib import Path
from typing import Any
import json
import re

import portalocker

from src.automation.artifact_exchange.archive_paths import update_state
from src.automation.artifact_exchange.core import component, locked
from src.automation.artifact_exchange.model_builds import PENDING, archived_builds, load
from src.shared.file_ops import atomic_json, fsync_dir, real_tree
from src.shared.model_provenance import PROPERTY_FILE, read_properties

ACTIONS = {'hide', 'show', 'archive', 'restore_archive', 'rename', 'renumber'}
MOVE_JOURNAL = 'model_move.json'


def inventory(library):
    from src.proj import PATH, MACHINE
    from src.res.model.util.core.model_path import source_models

    views = library._model_views()
    rows = {}

    def add(host, origin, name, build, path, archived, external, preview_only=False):
        model_id = library.model_id(dict(host=host, model=origin))
        key = sha256(json.dumps([model_id, build, str(path)]).encode()).hexdigest()
        rows[key] = dict(id=key, model_id=model_id, host=host, source_name=origin, name=name,
                         build=build, path=str(path), archived=archived, external=external,
                         preview_only=preview_only, hidden=views.get(model_id, {}).get(build, 'active') != 'active')

    for path, index in archived_builds(library.config):
        relative = path.relative_to(library.config.results / 'external/models')
        add(index['host'], index['model'], relative.parts[-2], path.name, path,
            len(relative.parts) == 4, True)
    for path, manifest in library._batches('model_training'):
        origin = manifest.get('experiment') or manifest['task']
        add(manifest.get('host') or 'unknown', origin, origin, '', path.parent, False, True, True)
    for model, path, stamp in source_models(PATH.model):
        build = stamp
        with suppress(OSError, ValueError, KeyError, TypeError):
            build = read_properties(path / PROPERTY_FILE).get('build', {}).get('name') or stamp
        add(library.config.source_host, model.full_name, model.full_name, build, path, bool(stamp),
            False, (path / '.build-preview-only').exists())
    from yaml import safe_load
    from src.res.model.util.core.basic import parse_model_input
    from src.res.model.util.core.model_path import ModelPath
    for row in rows.values():
        model = ModelPath(row['name'])
        row['short_name'] = model.model_name
        fields = parse_model_input(row['name'])
        row.update(module_type=fields['module_type'], module=fields['module_name'],
                   short='yes' if fields['st'] else 'no', stage='unknown', started='')
        row['host_display'] = row['host'] if row['external'] else MACHINE.name
        base = real_tree(row['path'])
        row['fingerprint'] = (load(base / 'build.json', {}).get('build', {}).get('fingerprint')
                              if row['external'] else read_properties(base / PROPERTY_FILE).get('build', {}).get('fingerprint'))
        with suppress(OSError, ValueError, KeyError, TypeError, AttributeError):
            row['started'] = read_properties(base / PROPERTY_FILE).get('initial_training', {}).get('started_at', '')
        with suppress(OSError, ValueError, KeyError, TypeError, AttributeError):
            row['stage'] = safe_load((base / 'configs/lifecycle.yaml').read_text()).get('stage', 'unknown')
    return rows


def _name(model_name, action, value):
    from src.res.model.util.core.basic import combine_full_name, parse_model_input

    fields = parse_model_input(model_name)
    fields.pop('full_name')
    if action == 'rename':
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}(@[1-9][0-9]{0,5})?', value):
            raise ValueError('Use 1–64 letters, digits, underscores, dots or hyphens for the model name')
        parts = value.rsplit('@', 1)
        if len(parts) == 2:
            if not parts[1].isdigit() or int(parts[1]) < 1:
                raise ValueError('Invalid model number')
            fields['model_name_index'] = int(parts[1])
        fields['model_clean_name'] = parts[0]
    elif action == 'renumber':
        if not isinstance(value, str) or not re.fullmatch(r'[1-9][0-9]{0,5}', value):
            raise ValueError('Enter a positive model number (1 means no suffix)')
        fields['model_name_index'] = int(value)
    return combine_full_name(**fields)


def _destination(library, row, action, value, reserved):
    from src.proj import PATH
    from src.res.model.util.core.model_path import ModelPath

    source = real_tree(row['path'])
    name = row['name']
    if action in ('hide', 'show'):
        return name, source
    if row['preview_only']:
        raise ValueError('This historical copy is for preview only')
    if action == 'archive' and row['archived'] or action == 'restore_archive' and not row['archived']:
        raise ValueError('The selected build is already in the requested archive state')
    if action in ('rename', 'renumber'):
        name = _name(name, action, value)
    if row['external']:
        root = library.config.results / 'external/models'
        relative = source.relative_to(root)
        host = relative.parts[-3]
        archived = action == 'archive' or (row['archived'] and action != 'restore_archive')
        parent = root / 'archive' if archived else root
        return name, real_tree(parent / host / component(name) / source.name)

    model = ModelPath(row['source_name'])
    if action == 'archive':
        marker = source / '.lifecycle.lock'
        stat = source.stat()
        seconds = getattr(stat, 'st_birthtime', None)
        if seconds is None:
            seconds = marker.stat().st_mtime if marker.exists() else datetime.now().timestamp()
        stamp = datetime.fromtimestamp(seconds)
        target = PATH.model_archive / f'{name}.{stamp:%Y%m%d%H%M%S}'
        while target.exists() or target in reserved:
            stamp += timedelta(seconds=1)
            target = PATH.model_archive / f'{name}.{stamp:%Y%m%d%H%M%S}'
        return name, real_tree(target)
    if action == 'restore_archive':
        index = model.find_new_index(folder_not_exist=True)
        candidate = model.with_new_index(index)
        while candidate.base.exists() or candidate.base in reserved:
            index += 1
            candidate = model.with_new_index(index)
        return candidate.full_name, real_tree(candidate.base)
    candidate = ModelPath(name)
    if row['archived']:
        return name, real_tree(source.with_name(f"{name}.{source.name.rsplit('.', 1)[1]}"))
    return name, real_tree(candidate.base)


def _hidden(library, row, name, target, hidden):
    path = library.state / 'model_views.json'
    views = library._model_views()
    key, build = row['model_id'], row['build']
    if key in views:
        views[key].pop(build, None)
        if not views[key]:
            views.pop(key)
    if hidden:
        if not row['external']:
            key = library.model_id(dict(host=row['host'], model=name))
            if not read_properties(target / PROPERTY_FILE).get('build', {}).get('name'):
                build = target.name.rsplit('.', 1)[1] if target.parent.name == 'archive' else ''
        views.setdefault(key, {})[build] = 'hidden'
    atomic_json(path, views)


def _finish(library, journal, operation):
    row, source, target, name, action = (operation[k] for k in ('row', 'source', 'target', 'name', 'action'))
    source, target = real_tree(source), real_tree(target)
    if source.exists() and target.exists():
        raise ValueError('Both model locations exist. Resolve the conflict before continuing.')
    if source.exists():
        from src.res.model.util.core.model_path import ModelPath

        target.parent.mkdir(parents=True, exist_ok=True)
        if not row['external'] and not row['archived'] and action in ('rename', 'renumber'):
            model, candidate = ModelPath(row['source_name']), ModelPath(name)
            model.rename(candidate.model_clean_name, candidate.model_name_index)
        else:
            source.rename(target)
        fsync_dir(target.parent)
        fsync_dir(source.parent)
    if not target.is_dir():
        raise ValueError('The planned model directory is unavailable')
    if row['external']:
        update_state(library.config, library.state.parent, {source: target})
    else:
        from src.res.model.util.core.model_path import ModelPath
        from src.proj import PATH

        if name != row['source_name']:
            candidate = ModelPath(name)
            for config_name in ('model', 'schedule'):
                config = target / 'configs' / (config_name + '.yaml')
                if config.is_file():
                    values = PATH.read_yaml(config)
                    if values.get('model.name') != candidate.model_clean_name:
                        PATH.dump_yaml(values | {'model.name': candidate.model_clean_name}, config, overwrite=True)
        if action == 'restore_archive':
            ModelPath(name).log_operation('resume_from_archive')
        elif action == 'archive':
            ModelPath(row['source_name']).log_operation('move_to_archive')
    _hidden(library, row, name, target, row['hidden'])
    if operation.get('history_id'):
        history = operation_history(library)
        batch = next(item for item in history if item['id'] == operation['history_id'])
        if not any(item['source'] == str(source) and item['target'] == str(target) for item in batch['changes']):
            batch['changes'].append(dict(row=row, name=name, source=str(source), target=str(target), hidden=row['hidden']))
            atomic_json(library.state / 'model_history.json', history)
    journal.unlink()
    fsync_dir(journal.parent)


def manage(library, action, items, *, apply=False, entry='web'):
    recovering = action == 'recover' and apply and items == []
    if not recovering and (action not in ACTIONS or not isinstance(items, list) or not 0 < len(items) <= 500):
        raise ValueError('Select up to 500 builds and a supported operation')
    if any(not isinstance(item, dict) or not isinstance(item.get('id'), str) for item in items):
        raise ValueError('Invalid build selection')
    if len({item['id'] for item in items}) != len(items):
        raise ValueError('Select each build only once')

    with ExitStack() as locks:
        # Receivers acquire this lock before any monitor operation locks.
        locks.enter_context(locked(library.config, 'receive'))
        locks.enter_context(portalocker.Lock(str(library.state / 'operations.lock'), timeout=2))
        if library.instances:
            locks.enter_context(portalocker.Lock(str(library.instances.state / 'operations.lock'), timeout=2))
        journal = library.state / MOVE_JOURNAL
        resumed = load(journal) if apply else None
        if resumed:
            pending = resumed
            source = real_tree(pending['source'])
            with ExitStack() as recovery:
                if not pending['row']['external'] and source.is_dir():
                    recovery.enter_context(portalocker.Lock(str(real_tree(source / '.lifecycle.lock')), timeout=0))
                _finish(library, journal, pending)
        if recovering:
            return [{'ok': True}] if resumed else []
        catalog, results, reserved = inventory(library), [], set()
        history_path = library.state / 'model_history.json'
        history = operation_history(library)
        batch: dict[str, Any] = dict(id=uuid4().hex, time=datetime.now(timezone.utc).isoformat(), entry=entry,
                     action=action, changes=[], undone=False)
        if apply:
            history.append(batch)
            atomic_json(history_path, history)
        for item in items:
            key = item['id']
            result = {'id': key}
            results.append(result)
            try:
                if resumed and resumed['row']['id'] == key and resumed['action'] == action:
                    result.update(ok=True, **{k: resumed[k] for k in ('name', 'source', 'target')})
                    continue
                row = catalog.get(key)
                if row is None:
                    raise ValueError('Model location changed. Refresh the list.')
                source = real_tree(row['path'])
                if not row['external'] and action == 'archive':
                    marker = real_tree(source / '.lifecycle.lock')
                    with suppress(FileExistsError):
                        marker.touch(exist_ok=False)
                name, target = _destination(library, row, action, item.get('value'), reserved)
                result.update(name=name, source=str(source), target=str(target))
                if target != source and (target.exists() or target in reserved):
                    raise ValueError('The destination already exists or is selected by another build')
                if apply and item.get('target') != str(target):
                    raise ValueError('The proposed destination changed. Review the operation again.')
                if action in ('hide', 'show'):
                    if apply:
                        _hidden(library, row, name, target, action == 'hide')
                elif target != source:
                    with ExitStack() as guards:
                        if row['external'] and (source / PENDING).exists():
                            raise ValueError('Finish receiving this model update before moving it')
                        if not row['external']:
                            guards.enter_context(portalocker.Lock(str(real_tree(source / '.lifecycle.lock')), timeout=0))
                        if library.instances and any(
                            real_tree(value).is_relative_to(source)
                            for instance in library.instances.discover()
                            for value in (instance.get('source'), instance.get('archive_source')) if value
                        ):
                            raise ValueError('Stop the running experiment for this model before moving it')
                        if apply:
                            operation = dict(row=row, name=name, source=str(source), target=str(target), action=action, history_id=batch['id'])
                            atomic_json(journal, operation)
                            _finish(library, journal, operation)
                if apply:
                    history = operation_history(library)
                    batch = next(item for item in history if item['id'] == batch['id'])
                if apply and action in ('hide', 'show') and row['hidden'] != (action == 'hide'):
                    batch['changes'].append(dict(row=row, name=name, source=str(source), target=str(target),
                                                 hidden=action == 'hide' if action in ('hide', 'show') else row['hidden']))
                    atomic_json(history_path, history)
                reserved.add(target)
                result['ok'] = True
            except (AssertionError, OSError, ValueError, KeyError, portalocker.exceptions.LockException) as exc:
                result['error'] = str(exc) or type(exc).__name__
                if apply and journal.exists():
                    break
        if not apply:
            grouped = {}
            for result in results:
                if result.get('source') and result['source'] != result.get('target'):
                    grouped.setdefault(result['target'], []).append(result)
            for group in grouped.values():
                if len(group) > 1:
                    for result in group:
                        result.pop('ok', None)
                        result['error'] = 'Multiple selected models share the destination'
        return results


def operation_history(library):
    return load(library.state / 'model_history.json', [])


def undo_latest(library, *, apply=False, expected=None):
    """Undo the latest batch, keeping files received since the original action."""
    with ExitStack() as locks:
        locks.enter_context(locked(library.config, 'receive'))
        locks.enter_context(portalocker.Lock(str(library.state / 'operations.lock'), timeout=2))
        history = operation_history(library)
        batch = next((item for item in reversed(history) if item['changes'] and not item['undone']), None)
        if batch is None:
            return {'changes': [], 'ok': False, 'error': 'No operation to undo'}
        if apply and expected is not None and batch['id'] != expected:
            raise ValueError('The latest operation changed. Review undo again.')
        pending_undo = library.state / 'undo_move.json'
        if apply and (pending := load(pending_undo)):
            _finish(library, pending_undo, pending)
            for change in batch['changes']:
                if change['source'] == pending['target'] and change['target'] == pending['source']:
                    change['undone'] = True
            atomic_json(library.state / 'model_history.json', history)
        preview = []
        rows = inventory(library)
        for change in reversed(batch['changes']):
            if change.get('undone'):
                continue
            source, target = real_tree(change['target']), real_tree(change['source'])
            original = change['row']
            current = next((row for row in rows.values() if row['path'] == str(source)
                            and row['build'] == original['build'] and row['host'] == original['host']
                            and row.get('fingerprint') == original.get('fingerprint')), None)
            error = 'Model identity changed' if current is None else ''
            if target != source and target.exists():
                error = 'The original path is occupied'
            preview.append(dict(source=str(source), target=str(target), error=error))
        if not apply or any(item['error'] for item in preview):
            return {'id': batch['id'], 'changes': preview, 'ok': not any(item['error'] for item in preview)}
        if library.instances:
            locks.enter_context(portalocker.Lock(str(library.instances.state / 'operations.lock'), timeout=2))
        for change in reversed(batch['changes']):
            if change.get('undone'):
                continue
            source, target = real_tree(change['target']), real_tree(change['source'])
            row = change['row']
            with ExitStack() as guards:
                if not row['external']:
                    guards.enter_context(portalocker.Lock(str(source / '.lifecycle.lock'), timeout=0))
                if (source / PENDING).exists():
                    raise ValueError('A model update is pending')
                if library.instances and any(real_tree(value).is_relative_to(source)
                    for instance in library.instances.discover()
                    for value in (instance.get('source'), instance.get('archive_source')) if value):
                    raise ValueError('Stop the running experiment before undoing')
                if source != target:
                    operation = dict(row=row | {'path': str(source), 'source_name': change['name']},
                                     name=row['name'], source=str(source), target=str(target), action='undo')
                    _finish(library, library.state / 'undo_move.json', _save_undo(library, operation))
                _hidden(library, row | {'model_id': library.model_id(dict(host=row['host'],
                        model=row['source_name'] if row['external'] else change['name']))},
                        row['name'], target, row['hidden'])
                change['undone'] = True
                atomic_json(library.state / 'model_history.json', history)
        batch['undone'] = True
        batch['undone_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(library.state / 'model_history.json', history)
        return {'ok': True, 'changes': preview}


def _save_undo(library, operation):
    atomic_json(library.state / 'undo_move.json', operation)
    return operation


def cli_manage(paths, action, value=None):
    from .artifact_library import Library
    from src.proj import Logger

    library = Library('model_preview')
    rows = inventory(library)
    selected = [dict(id=row['id'], value=value) for row in rows.values()
                if not row['external'] and Path(row['path']) in paths]
    if not selected:
        return
    plan = manage(library, action, selected, entry='cli')
    values = {item['id']: item for item in selected}
    valid = [item | {'value': values[item['id']]['value']} for item in plan if item.get('ok')]
    Logger.stdout_pairs({item['source']: item.get('error', item['target']) for item in plan},
                        title='Model operation preview')
    if valid and input('Apply these model operations? [y/N] ').strip().lower() == 'y':
        result = manage(library, action, valid, apply=True, entry='cli')
        Logger.stdout_pairs({item['id']: item.get('error', 'Done') for item in result}, title='Model operations')
