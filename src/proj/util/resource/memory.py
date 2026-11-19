"""Process memory measurements without tensor or data dependencies."""
import os
import subprocess
from pathlib import Path

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
