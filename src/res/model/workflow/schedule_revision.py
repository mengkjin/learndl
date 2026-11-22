"""Preserve ledger identity for the explicitly recorded schedule layout move."""
from __future__ import annotations

import json
from pathlib import Path

from src.proj import PATH
from src.res.model.util.training_history import file_revision, git


def schedule_revision(path: Path) -> dict:
    revision = file_revision(path)
    root = PATH.main.resolve()
    manifest = root / 'configs/schedule/layout_migration.json'
    if not manifest.is_file() or not path.resolve().is_relative_to(root):
        return revision
    relative = path.resolve().relative_to(root).as_posix()
    migration = json.loads(manifest.read_text())
    move = migration['moves'].get(relative)
    if not move or move['sha256'] != revision['sha256']:
        return revision

    commit = revision['git_commit']
    parent = git('rev-parse', f'{commit}^') if commit else None
    if parent and git('rev-parse', '--verify', f'{parent}:{relative}'):
        return revision  # Subsequent edit, even if it restores the original bytes.

    old = move['old']
    baseline = migration['base']
    expected = git('rev-parse', '--verify', f'{baseline}:{old}')
    before = git('rev-parse', '--verify', f'{parent}:{old}') if parent else None
    after = git('rev-parse', '--verify', f'{commit}:{relative}') if commit else None
    if not parent or not expected or before != expected or after != expected or git(
            'rev-parse', '--verify', f'{commit}:{old}'):
        raise ValueError(f'Cannot verify schedule layout migration for {relative}; check Git history')
    previous_commit = git('log', '-1', '--format=%H', parent, '--', old)
    if not previous_commit:
        raise ValueError(f'Missing pre-move Git revision for {relative}')
    return revision | {'path': str(root / old), 'git_commit': previous_commit}
