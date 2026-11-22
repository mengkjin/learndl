"""Validate names and calculate model operation destinations."""
from __future__ import annotations

from datetime import datetime, timedelta
import re

from src.automation.artifact_exchange.core import component
from src.shared.file_ops import real_tree

__all__ = ['ACTIONS', 'destination']

ACTIONS = {'hide', 'show', 'archive', 'restore_archive', 'rename', 'renumber'}


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


def destination(context, row, action, value, reserved):
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
        root = context.config.results / 'external/models'
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
            seconds = marker.stat().st_mtime if marker.exists() else stat.st_mtime
        stamp = datetime.fromtimestamp(seconds)
        target = context.model_archive / f'{name}.{stamp:%Y%m%d%H%M%S}'
        while target.exists() or target in reserved:
            stamp += timedelta(seconds=1)
            target = context.model_archive / f'{name}.{stamp:%Y%m%d%H%M%S}'
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
