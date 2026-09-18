"""Preflight large CPU allocations against physical and cgroup headroom."""
from __future__ import annotations

from pathlib import Path

import psutil


def available_bytes() -> int:
    available = int(psutil.virtual_memory().available)
    try:
        lines = Path('/proc/self/cgroup').read_text().splitlines()
    except OSError:
        return available
    root = Path('/sys/fs/cgroup').resolve()
    for line in lines:
        if not line.startswith('0::'):
            continue
        group = (root / line[3:].lstrip('/')).resolve()
        for path in (group, *group.parents):
            if not path.is_relative_to(root):
                break
            try:
                maximum = (path / 'memory.max').read_text().strip()
                current = int((path / 'memory.current').read_text())
                if maximum != 'max':
                    available = min(available, max(0, int(maximum) - current))
            except (OSError, ValueError):
                continue
    return available


def check_allocation(size: int, label: str, *, limit: int = 32 * 1024**3) -> None:
    """Fail before allocation; this is a conservative preflight, not an OOM guarantee."""
    budget = min(limit, int(available_bytes() * .8))
    if size > budget:
        raise MemoryError(
            f'{label}: refusing {size / 1024**3:.2f} GiB allocation; '
            f'budget={budget / 1024**3:.2f} GiB (80% of current system/cgroup headroom, '
            f'capped at {limit / 1024**3:.0f} GiB). Existing tensors are still live. '
            'Inspect the secid universe and task memory log before retrying.'
        )
