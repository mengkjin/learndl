"""Temporary RSS / object-size probes for large PrePro jobs.

Enable with ``PreProcessor.MemTrace = True`` (on for ``minc`` / ``mincr``).
Remove once year-chunk RAM is confirmed stable.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any , Callable , cast

import numpy as np
import torch

__all__ = ['fmt_bytes', 'rss_bytes', 'hwm_bytes', 'nbytes_of', 'describe', 'mesh4d_bytes', 'log_mem']


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


def mesh4d_bytes(n : int , t : int , i : int , f : int) -> int:
    """Logical bytes if eight int64 grids were materialized, NOT actual storage."""
    return int(n) * int(t) * int(i) * int(f) * 8 * 8


def nbytes_of(obj : Any) -> int:
    """Best-effort payload size of a named object."""
    if obj is None:
        return 0
    if isinstance(obj , (int , float , np.integer , np.floating)):
        return int(obj)
    if isinstance(obj , torch.Tensor):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj , np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj , (list , tuple)):
        return sum(nbytes_of(x) for x in obj)
    if isinstance(obj , dict):
        return sum(nbytes_of(v) for v in obj.values())
    values = getattr(obj , 'values' , None)
    if isinstance(values , torch.Tensor) and hasattr(obj , 'shape'):
        n = int(values.numel() * values.element_size())
        for name in ('secid' , 'date' , 'feature'):
            arr = getattr(obj , name , None)
            if arr is not None:
                n += int(np.asarray(arr).nbytes)
        return n
    estimated_size = getattr(obj , 'estimated_size' , None)
    if callable(estimated_size):
        return int(cast(Callable[[] , int] , estimated_size)())
    memory_usage = getattr(obj , 'memory_usage' , None)
    if callable(memory_usage):
        usage = cast(Any , memory_usage)(deep = True)
        return int(usage.sum())
    return 0


def describe(obj : Any) -> str:
    """One-token size summary, with shape/dtype for tensors and DataBlocks."""
    if obj is None:
        return '-'
    if isinstance(obj , (int , float , np.integer , np.floating)):
        return fmt_bytes(int(obj))
    if isinstance(obj , torch.Tensor):
        return f'logical={fmt_bytes(nbytes_of(obj))},storage={fmt_bytes(obj.untyped_storage().nbytes())}{tuple(obj.shape)},{obj.dtype}'
    if isinstance(obj , np.ndarray):
        return f'{fmt_bytes(obj.nbytes)}{obj.shape},{obj.dtype}'
    values = getattr(obj , 'values' , None)
    if isinstance(values , torch.Tensor) and hasattr(obj , 'shape'):
        return f'logical={fmt_bytes(nbytes_of(obj))},storage={fmt_bytes(values.untyped_storage().nbytes())}{tuple(obj.shape)},{values.dtype}'
    if hasattr(obj , 'estimated_size') and hasattr(obj , 'shape'):
        rows , cols = obj.shape
        return f'{fmt_bytes(nbytes_of(obj))}pl({rows},{cols})'
    if isinstance(obj , (list , tuple)):
        return f'{fmt_bytes(nbytes_of(obj))}n={len(obj)}'
    return fmt_bytes(nbytes_of(obj))


def log_mem(logger : Any , tag : str , **objects : Any) -> None:
    """Write RSS (and optional HWM) plus named object sizes at verbosity 2."""
    parts = [f'time={datetime.now().astimezone().isoformat(timespec="milliseconds")}',
             f'pid={os.getpid()}', f'rss={fmt_bytes(rss_bytes())}']
    hwm = hwm_bytes()
    if hwm is not None:
        parts.append(f'hwm={fmt_bytes(hwm)}')
    parts.extend(f'{"mesh_logical_estimate" if name == "mesh4d" else name}={describe(obj)}' for name , obj in objects.items())
    logger.stdout(f'mem {tag}: ' + ' | '.join(parts) , vb = 2 , add_prefix = False)


def trace_stage(tag: str, **objects: Any) -> None:
    """Opt-in probes for shared DataBlock operations, enabled by CLI reconstruction."""
    if os.environ.get('LEARNDL_MEMORY_TRACE') == '1':
        log_mem(_DiagnosticLog, tag, **objects)


class _DiagnosticLog:
    @staticmethod
    def stdout(message: str, **kwargs: Any) -> None:
        # Explicit diagnostics must survive normal verbosity filtering. The
        # reconstruction's tee still captures this without buffering in memory.
        print(message, flush=True)
