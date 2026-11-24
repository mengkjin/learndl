# coding: utf-8
# description: Move legacy HTML captures into the active models' logs directories.
# content: Back up legacy captures and migrate them to logs/html for model metadata publication.
# email: False
# mode: os

"""Migrate HTML captures for active models while preserving model archives."""
from __future__ import annotations

from src.automation.artifact_exchange.model_metadata import migrate_model_html
from src.shared.file_ops import real_tree


def _legacy_directories(root):
    for family in ('nn', 'boost', 'linear', 'factor', 'st'):
        for directory in sorted(real_tree(root / family).glob('*/run_logs/html')):
            directory = real_tree(directory)
            if directory.is_dir():
                yield directory


def apply(ctx):
    from src.proj import Logger, PATH

    directories = list(_legacy_directories(PATH.model))
    for directory in directories:
        backup = ctx.backup_dir / 'models' / directory.relative_to(PATH.model)
        migrate_model_html(directory.parent.parent, backup_dir=backup)
    Logger.stdout(f'Checked {len(directories)} active model directories for legacy HTML logs')


def verify(ctx):
    from src.proj import PATH

    return not any(real_tree(path).is_file() for directory in _legacy_directories(PATH.model)
                   for path in directory.glob('*.html'))
