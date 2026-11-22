"""Memory allocation policy for data block processing."""
from __future__ import annotations

from src.shared.resource.memory import allocation_budget, available_bytes

__all__ = ['check_allocation']


def check_allocation(size: int, label: str, *, limit: int | None = 32 * 1024**3) -> None:
    """Fail before allocation; this is a conservative preflight, not an OOM guarantee."""
    budget = allocation_budget(available_bytes(), fraction=.8, limit=limit or None)
    if size > budget:
        raise MemoryError(
            f'{label}: refusing {size / 1024**3:.2f} GiB allocation; '
            f'budget={budget / 1024**3:.2f} GiB (80% of current system/cgroup headroom, '
            f'capped at {limit / 1024**3:.0f} GiB). Existing tensors are still live. ' if limit else ''
            'Inspect the secid universe and task memory log before retrying.'
        )
