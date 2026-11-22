"""Process memory measurements without tensor or data dependencies."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

__all__ = ['fmt_bytes', 'rss_bytes', 'hwm_bytes', 'available_bytes', 'allocation_budget']

def fmt_bytes(n : int | float) -> str:
    """Format a byte count as ``12.3G`` / ``512M``."""
    x = float(n)
    for unit in ('B' , 'K' , 'M' , 'G' , 'T'):
        if abs(x) < 1024.0 or unit == 'T':
            return f'{int(x)}B' if unit == 'B' else f'{x:.1f}{unit}'
        x /= 1024.0
    return f'{x:.1f}T'


def rss_bytes() -> int:
    """Current process RSS. Linux ``VmRSS``, otherwise ``ps -o rss`` (KB)."""
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    out = subprocess.check_output(['ps' , '-o' , 'rss=' , '-p' , str(os.getpid())] , text = True)
    return int(out.strip().split()[-1]) * 1024


def hwm_bytes() -> int | None:
    """Peak RSS from Linux ``VmHWM``, else None."""
    status = Path('/proc/self/status')
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    return None


def available_bytes() -> int:
    import psutil

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


def allocation_budget(available: int, *, fraction: float, limit: int | None) -> int:
    """Compute a budget using the caller's fraction of available bytes and optional cap."""
    budget = int(available * fraction)
    return budget if limit is None else min(budget, limit)
