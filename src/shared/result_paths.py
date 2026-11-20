"""Paths for project-level research and operational output, without project initialization."""
from __future__ import annotations

from pathlib import Path

__all__ = ['custom_results', 'storage_reports', 'relocated_result']


def custom_results(project_root: Path) -> Path:
    return Path(project_root) / 'results' / 'custom'


def storage_reports(project_root: Path) -> Path:
    return custom_results(project_root) / 'storage_reports'


def relocated_result(path, project_root, category):
    """Resolve a stored pre-0.7.4 path without editing the signed input document."""
    root = Path(project_root).resolve()
    path = (root / Path(path).expanduser()).resolve()
    old = root / 'results' / category
    if path.is_relative_to(old):
        return (custom_results(root) / category / path.relative_to(old)).resolve()
    return path
