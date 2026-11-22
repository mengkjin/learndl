"""Schedule discovery without importing the model training stack."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .path import PATH

__all__ = ['find_schedule_path', 'available_schedules']


def schedule_roots() -> tuple[Path, ...]:
    return PATH.sched, PATH.sched_shared, PATH.sched_linear


def schedule_stem(name: str) -> str:
    parts = name.split('/')
    if ('\\' in name or ':' in name or '\x00' in name or name.endswith('.yaml')
            or any(not part or part in ('.', '..') for part in parts)):
        raise ValueError(f'Invalid schedule reference: {name!r}')
    return parts[-1]


def _files(root: Path):
    for path in root.rglob('*.yaml'):
        if path.is_file() and path.resolve().is_relative_to(root.resolve()):
            yield path


def schedule_matches(name: str, roots: Sequence[Path] | None = None) -> list[Path]:
    leaf = schedule_stem(name) + '.yaml'
    roots = schedule_roots() if roots is None else roots
    candidates = []
    for root in roots:
        if '/' in name:
            path = root / f'{name}.yaml'
            if path.is_file():
                if not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(f'Schedule escapes its root: {path}')
                candidates.append(path)
        else:
            for path in root.rglob('*.yaml'):
                if path.name == leaf and path.is_file():
                    if not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError(f'Schedule escapes its root: {path}')
                    candidates.append(path)
    return sorted(set(candidates))


def find_schedule_path(name: str | None, *, roots: Sequence[Path] | None = None) -> Path | None:
    if not name:
        return None
    paths = schedule_matches(name, roots)
    if len(paths) > 1:
        raise ValueError(f'Duplicate schedule {name}: ' + ', '.join(map(str, paths)))
    return paths[0] if paths else None


def schedule_source(path: Path) -> str:
    for label, root in zip(('current', 'shared', 'linear'), schedule_roots()):
        if path.is_relative_to(root):
            return f'{label}/{path.relative_to(root).with_suffix("").as_posix()}'
    raise ValueError(f'Schedule outside configured roots: {path}')


def is_linear_schedule(name: str | None) -> bool:
    path = find_schedule_path(name)
    return path is not None and path.is_relative_to(PATH.sched_linear)


def available_schedules(*, roots: Sequence[Path] | None = None) -> list[str]:
    """List relative references and reject references that cannot be selected."""
    selected = schedule_roots() if roots is None else roots
    names = sorted({path.relative_to(root).with_suffix('').as_posix()
                    for root in selected for path in _files(root)})
    for name in names:
        find_schedule_path(name)
    return names
