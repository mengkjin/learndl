"""Lightweight process/cgroup sampler; also executable directly without project imports."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import psutil


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def identity(pid: int) -> dict:
    process = psutil.Process(pid)
    return {'pid': pid, 'created': process.create_time()}


def alive(process: dict) -> bool:
    try:
        actual = psutil.Process(process['pid'])
        return actual.create_time() == process['created'] and actual.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError, TypeError):
        return False


def cgroup_path(pid: int) -> Path | None:
    try:
        for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                root = Path('/sys/fs/cgroup').resolve()
                path = (root / line[3:].lstrip('/')).resolve()
                return path if path.is_relative_to(root) else None
    except OSError:
        pass
    return None


def read_cgroup(path: Path | None) -> dict:
    result = {}
    if path is not None:
        # Parent limits can be tighter than the process's own leaf limit.
        for parent in (path, *path.parents):
            if not parent.is_relative_to('/sys/fs/cgroup'):
                break
            fields = {}
            for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.high',
                         'memory.swap.current', 'memory.swap.max', 'memory.events', 'memory.pressure'):
                try:
                    fields[name] = (parent / name).read_text().strip()
                except OSError as exc:
                    fields[name] = f'unavailable: {exc}'
            result[str(parent)] = fields
    return result


def sample(process: dict, group: Path | None) -> dict:
    rows = []
    if alive(process):
        try:
            root = psutil.Process(process['pid'])
            for item in [root, *root.children(recursive=True)]:
                try:
                    rows.append({'pid': item.pid, 'rss': item.memory_info().rss, 'name': item.name()})
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {'time': time.time(), 'processes': rows, 'tree_rss': sum(row['rss'] for row in rows),
            'available': memory.available, 'total': memory.total, 'swap_used': swap.used,
            'swap_total': swap.total, 'cgroups': read_cgroup(group)}


def evidence(started: float, process: dict, group: Path | None) -> dict:
    result = {'time': time.time(), 'process': process, 'cgroups': read_cgroup(group),
              'interpretation': 'Unexpected termination is not proof of OOM. Correlate PID, time and cgroup.'}
    # Bounded output/time. Permission failures are evidence gaps, never "no OOM".
    for name, options in (('kernel', ['-k']), ('oomd', ['-u', 'systemd-oomd.service'])):
        try:
            output = subprocess.run(['journalctl', *options, '--since', f'@{int(started)}',
                                     '--until', f'@{int(time.time())}', '-n', '200', '--no-pager',
                                     '--output=short-iso'], capture_output=True, text=True, timeout=5)
            result[name] = {'returncode': output.returncode, 'stdout': output.stdout[-32768:],
                            'stderr': output.stderr[-4096:]}
        except (OSError, subprocess.SubprocessError) as exc:
            result[name] = {'unavailable': str(exc)}
    return result


def run(directory: Path, process: dict, interval: float = 2) -> None:
    started = time.time()
    group = cgroup_path(process['pid'])
    atomic_json(directory / 'sampler.json', {'sampler': identity(os.getpid()), 'target': process,
                                            'cgroup': str(group) if group else None})
    with (directory / 'memory.jsonl').open('a', buffering=1) as output:
        while True:
            output.write(json.dumps(sample(process, group)) + '\n')
            if (directory / 'finished.json').exists():
                return
            if not alive(process):
                atomic_json(directory / 'evidence.json', evidence(started, process, group))
                return
            time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('pid', type=int)
    parser.add_argument('created', type=float)
    args = parser.parse_args()
    run(args.directory, {'pid': args.pid, 'created': args.created})
