"""Host-wide resource sampling; no CUDA context or task lifecycle writes."""
from __future__ import annotations

import csv
import io
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class GPUUsage:
    index: str
    name: str
    used_mib: float | None
    total_mib: float | None
    utilization: float | None

    @property
    def memory_percent(self) -> float | None:
        if self.used_mib is None or not self.total_mib:
            return None
        return self.used_mib / self.total_mib * 100


@dataclass(frozen=True)
class ResourceSnapshot:
    sampled_at: float
    memory: tuple[float, float, float] | None
    cpu_percent: float | None
    disk: tuple[float, float, float] | None
    gpus: tuple[GPUUsage, ...]
    errors: tuple[str, ...]


def _number(value: str, *, percent: bool = False) -> float | None:
    try:
        result = float(value.strip())
        if not math.isfinite(result) or result < 0 or (percent and result > 100):
            return None
        return result
    except ValueError:
        return None


def sample_resources(project_path: Path) -> ResourceSnapshot:
    errors: list[str] = []
    memory = disk = None
    cpu = None
    try:
        mem = psutil.virtual_memory()
        memory = (mem.total - mem.available, mem.total, mem.percent)
    except (OSError, psutil.Error) as exc:
        errors.append(f'Memory unavailable: {exc}')
    try:
        # A short real sample avoids reporting the meaningless first nonblocking 0%.
        cpu = psutil.cpu_percent(interval=0.1)
    except (OSError, psutil.Error) as exc:
        errors.append(f'CPU unavailable: {exc}')
    try:
        usage = psutil.disk_usage(str(project_path))
        disk = (usage.used, usage.total, usage.percent)
    except (OSError, psutil.Error) as exc:
        errors.append(f'Disk unavailable: {exc}')
    gpus: list[GPUUsage] = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2, check=True,
        )
        for row in csv.reader(io.StringIO(result.stdout)):
            if not row:
                continue
            if len(row) != 5:
                errors.append('GPU sample contained an unreadable device row.')
                continue
            index, name, used, total, utilization = (value.strip() for value in row)
            gpus.append(GPUUsage(index, name, _number(used), _number(total),
                                 _number(utilization, percent=True)))
        if not gpus:
            errors.append('GPU metrics unavailable: no NVIDIA devices reported.')
    except FileNotFoundError:
        errors.append('GPU metrics unavailable: nvidia-smi is not installed on this host.')
    except subprocess.TimeoutExpired:
        errors.append('GPU metrics unavailable: nvidia-smi timed out after 2 seconds.')
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f'GPU metrics unavailable: {exc}')
    return ResourceSnapshot(time.time(), memory, cpu, disk, tuple(gpus), tuple(errors))
