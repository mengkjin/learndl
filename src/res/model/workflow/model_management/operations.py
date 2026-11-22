"""Execute, journal and undo model operations under the existing runtime locks."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import portalocker

from src.automation.artifact_exchange.archive_paths import update_state
from src.automation.artifact_exchange.core import locked
from src.automation.artifact_exchange.model_builds import PENDING, load
from src.shared.file_ops import atomic_json, fsync_dir, real_tree
from src.shared.model_provenance import PROPERTY_FILE, model_id, read_properties

from .inventory import inventory, model_views
from .planning import ACTIONS, destination

__all__ = ['MOVE_JOURNAL', 'manage', 'operation_history', 'undo_latest']

MOVE_JOURNAL = 'model_move.json'


def _fallback_build(context, path):
    return path.name.rsplit('.', 1)[-1] if path.parent == context.model_archive else ''


def _hidden(context, row, name, target, hidden):
    path = context.state / 'model_views.json'
    views = model_views(context.state)
    key, build = row['model_id'], row['build']
    if key in views:
        views[key].pop(build, None)
        if not views[key]:
            views.pop(key)
    if hidden:
        if not row['external']:
            key = model_id(dict(host=row['host'], model=name))
            if not read_properties(target / PROPERTY_FILE).get('build', {}).get('name'):
                build = _fallback_build(context, target)
        views.setdefault(key, {})[build] = 'hidden'
    atomic_json(path, views)


def _finish(context, journal, operation):
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
        update_state(context.config, context.state.parent, {source: target})
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
    _hidden(context, row, name, target, row['hidden'])
    if operation.get('history_id'):
        history = operation_history(context)
        batch = next(item for item in history if item['id'] == operation['history_id'])
        if not any(item['source'] == str(source) and item['target'] == str(target) for item in batch['changes']):
            batch['changes'].append(dict(row=row, name=name, source=str(source), target=str(target), hidden=row['hidden']))
            atomic_json(context.state / 'model_history.json', history)
    if action != 'undo':
        journal.unlink()
        fsync_dir(journal.parent)


def manage(context, action, items, *, apply=False, entry='web'):
    recovering = action == 'recover' and apply and items == []
    if not recovering and (action not in ACTIONS or not isinstance(items, list) or not 0 < len(items) <= 500):
        raise ValueError('Select up to 500 builds and a supported operation')
    if any(not isinstance(item, dict) or not isinstance(item.get('id'), str) for item in items):
        raise ValueError('Invalid build selection')
    if len({item['id'] for item in items}) != len(items):
        raise ValueError('Select each build only once')

    with ExitStack() as locks:
        # Receivers acquire this lock before any monitor operation locks.
        locks.enter_context(locked(context.config, 'receive'))
        locks.enter_context(portalocker.Lock(str(context.state / 'operations.lock'), timeout=2))
        if context.instance_lock is not None:
            locks.enter_context(portalocker.Lock(str(context.instance_lock), timeout=2))
        journal = context.state / MOVE_JOURNAL
        if not recovering and (context.state / 'undo_move.json').exists():
            raise ValueError('Complete the interrupted undo before starting another operation')
        resumed = load(journal) if apply else None
        if resumed:
            pending = resumed
            source = real_tree(pending['source'])
            with ExitStack() as recovery:
                if not pending['row']['external'] and source.is_dir():
                    recovery.enter_context(portalocker.Lock(str(real_tree(source / '.lifecycle.lock')), timeout=0))
                _finish(context, journal, pending)
        if recovering:
            return [{'ok': True}] if resumed else []
        catalog, results, reserved = inventory(context), [], set()
        history_path = context.state / 'model_history.json'
        history = operation_history(context)
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
                name, target = destination(context, row, action, item.get('value'), reserved)
                result.update(name=name, source=str(source), target=str(target))
                if target != source and (target.exists() or target in reserved):
                    raise ValueError('The destination already exists or is selected by another build')
                if apply and item.get('target') != str(target):
                    raise ValueError('The proposed destination changed. Review the operation again.')
                if action in ('hide', 'show'):
                    if apply:
                        _hidden(context, row, name, target, action == 'hide')
                elif target != source:
                    with ExitStack() as guards:
                        if row['external'] and (source / PENDING).exists():
                            raise ValueError('Finish receiving this model update before moving it')
                        if not row['external']:
                            marker = real_tree(source / '.lifecycle.lock')
                            if apply or marker.exists():
                                guards.enter_context(portalocker.Lock(str(marker), timeout=0))
                        if context.running_instances and any(
                            real_tree(value).is_relative_to(source)
                            for instance in context.running_instances()
                            for value in (instance.get('source'), instance.get('archive_source')) if value
                        ):
                            raise ValueError('Stop the running experiment for this model before moving it')
                        if apply:
                            operation = dict(row=row, name=name, source=str(source), target=str(target), action=action, history_id=batch['id'])
                            atomic_json(journal, operation)
                            _finish(context, journal, operation)
                if apply:
                    history = operation_history(context)
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


def operation_history(context):
    return load(context.state / 'model_history.json', [])


def undo_latest(context, *, apply=False, expected=None):
    """Undo the latest batch, keeping files received since the original action."""
    with ExitStack() as locks:
        locks.enter_context(locked(context.config, 'receive'))
        locks.enter_context(portalocker.Lock(str(context.state / 'operations.lock'), timeout=2))
        history = operation_history(context)
        batch = next((item for item in reversed(history) if item['changes'] and not item['undone']), None)
        if batch is None:
            return {'changes': [], 'ok': False, 'error': 'No operation to undo'}
        if apply and expected is not None and batch['id'] != expected:
            raise ValueError('The latest operation changed. Review undo again.')
        pending_undo = context.state / 'undo_move.json'
        pending = load(pending_undo)
        if (context.state / MOVE_JOURNAL).exists():
            raise ValueError('Recover the interrupted model operation before undoing')
        preview = []
        rows = inventory(context)
        for change in reversed(batch['changes']):
            if change.get('undone'):
                continue
            source, target = real_tree(change['target']), real_tree(change['source'])
            original = change['row']
            if pending and pending['source'] == str(source) and pending['target'] == str(target):
                preview.append(dict(source=str(source), target=str(target), error=''))
                continue
            build = original['build']
            if not original['external'] and build == _fallback_build(context, target):
                build = _fallback_build(context, source)
            current = next((row for row in rows.values() if row['path'] == str(source)
                            and row['build'] == build and row['host'] == original['host']
                            and row.get('fingerprint') == original.get('fingerprint')), None)
            error = 'Model identity changed' if current is None else ''
            if target != source and target.exists():
                error = 'The original path is occupied'
            preview.append(dict(source=str(source), target=str(target), error=error))
        if not apply or any(item['error'] for item in preview):
            return {'id': batch['id'], 'changes': preview, 'ok': not any(item['error'] for item in preview)}
        if context.instance_lock is not None:
            locks.enter_context(portalocker.Lock(str(context.instance_lock), timeout=2))
        for change in reversed(batch['changes']):
            if change.get('undone'):
                continue
            source, target = real_tree(change['target']), real_tree(change['source'])
            row = change['row']
            replay = pending and pending['source'] == str(source) and pending['target'] == str(target)
            location = target if replay and not source.exists() else source
            with ExitStack() as guards:
                if not row['external']:
                    guards.enter_context(portalocker.Lock(str(real_tree(location / '.lifecycle.lock')), timeout=0))
                if (location / PENDING).exists():
                    raise ValueError('A model update is pending')
                if context.running_instances and any(real_tree(value).is_relative_to(location)
                    for instance in context.running_instances()
                    for value in (instance.get('source'), instance.get('archive_source')) if value):
                    raise ValueError('Stop the running experiment before undoing')
                current_row = row | {'model_id': model_id(dict(host=row['host'],
                    model=row['source_name'] if row['external'] else change['name']))}
                if not row['external'] and row['build'] == _fallback_build(context, target):
                    current_row['build'] = _fallback_build(context, source)
                if source != target:
                    operation = dict(row=current_row | {'path': str(source), 'source_name': change['name']},
                                     name=row['name'], source=str(source), target=str(target), action='undo')
                    atomic_json(pending_undo, operation)
                    _finish(context, pending_undo, operation)
                else:
                    _hidden(context, current_row, row['name'], target, row['hidden'])
                change['undone'] = True
                atomic_json(context.state / 'model_history.json', history)
                if source != target:
                    pending_undo.unlink()
                    fsync_dir(pending_undo.parent)
        # A crash after the last history write can leave the completed journal.
        if pending_undo.exists():
            pending_undo.unlink()
            fsync_dir(pending_undo.parent)
        batch['undone'] = True
        batch['undone_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(context.state / 'model_history.json', history)
        return {'ok': True, 'changes': preview}
