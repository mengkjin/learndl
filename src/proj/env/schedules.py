"""Schedule discovery without importing the model training stack."""
from __future__ import annotations

__all__ = [
    'find_schedule_path',
    'available_schedules',
]

from pathlib import Path
from collections.abc import Sequence

from .path import PATH


def find_schedule_path(name: str | None, *, roots: Sequence[Path] | None = None) -> Path | None:
    """Resolve a named schedule and reject ambiguous names across schedule roots."""
    if not name:
        return None
    if roots is None:
        roots = (PATH.sched, PATH.sched_shared, PATH.sched_linear)
    candidates = [root / f'{name}.yaml' for root in roots]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f'Duplicate schedule {name}: {existing}')
    return existing[0] if existing else None


def available_schedules() -> list[str]:
    """List schedules in stable order, validating duplicates before selection."""
    names = sorted({p.stem for root in (PATH.sched, PATH.sched_shared, PATH.sched_linear)
                    for p in root.glob('*.yaml')})
    for name in names:
        find_schedule_path(name)
    return names
