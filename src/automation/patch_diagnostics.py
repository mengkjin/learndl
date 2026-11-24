"""Bounded failure diagnostics for current and historical migration attempts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

from src.shared.file_ops import atomic_json

__all__ = ['check_open_files', 'failure_summary']

_EXCEPTION = re.compile(r'^[\w.]+(?:Error|Exception)(?::|$)')


_MOUNT_WARNING = re.compile(
    r"lsof: WARNING: can't stat\(\) fuse\.(?:portal|gvfsd-fuse) file system (/.+)"
)


def _probe_errors(stderr: str, path: Path) -> str:
    """Remove paired warnings for unrelated Linux desktop mount points."""
    lines = [line for line in stderr.splitlines() if line.strip()]
    if not lines:
        return ''
    try:
        selected = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return stderr.strip()
    errors = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _MOUNT_WARNING.fullmatch(line)
        if match and index + 1 < len(lines) and lines[index + 1].strip() == 'Output information may be incomplete.':
            mount = Path(match[1])
            if (mount.anchor == '/' and '\\' not in match[1] and '..' not in mount.parts
                    and not selected.is_relative_to(mount)
                    and not mount.is_relative_to(selected)):
                index += 2
                continue
        errors.append(line)
        index += 1
    return '\n'.join(errors)


def check_open_files(path: Path, *, timeout: int = 60) -> None:
    """Fail closed on occupied paths and on inconclusive probes."""
    args = ['+D', str(path)] if path.is_dir() else ['--', str(path)]
    try:
        probe = subprocess.run(['lsof', '-nP', '-Fpcn', *args], capture_output=True,
                               text=True, errors='replace', timeout=timeout, check=False,
                               env={**os.environ, 'LC_ALL': 'C'})
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'Could not check open files at {path}: {exc}') from exc
    errors = _probe_errors(probe.stderr, path)
    if errors or probe.returncode not in (0, 1):
        detail = errors[:1000] or f'lsof exited {probe.returncode}'
        raise RuntimeError(f'Could not check open files at {path}: {detail}')
    if probe.stdout:
        pid, command, holders = '', '', []
        for line in probe.stdout.splitlines():
            if line.startswith('p'):
                pid, command = line[1:], ''
            elif line.startswith('c'):
                command = line[1:]
            elif line.startswith('n') and pid and len(holders) < 10:
                holders.append(f'PID {pid} ({command or "unknown"}), {line[1:]}')
        if holders:
            raise RuntimeError(f'Open files at {path}: ' + ' | '.join(holders))
        raise RuntimeError(f'Could not check open files at {path}: unrecognized lsof output')
    if probe.returncode == 0:
        raise RuntimeError(f'Could not check open files at {path}: lsof reported success without a file')


def _blocked(report):
    if not isinstance(report, dict):
        return
    for field in ('moves', 'targets', 'backfill', 'entries', 'rows'):
        rows = report.get(field, [])
        if isinstance(rows, dict):
            rows = [dict(row, item=key) for key, row in rows.items() if isinstance(row, dict)]
        if isinstance(rows, list):
            yield from (row for row in rows if isinstance(row, dict) and (row.get('error') or row.get('reason'))
                        and (row.get('state') in ('failed', 'blocked') or row.get('status') in ('failed', 'blocked')))


def failure_summary(attempt, *, persist=False):
    """Read an existing summary or derive one for a legacy attempt."""
    log = Path(attempt['log_path'])
    backup = log.parent / 'backup'
    destination = log.parent / 'failure.json'
    try:
        return json.loads(destination.read_text())
    except (OSError, ValueError):
        pass

    error = attempt.get('error') or 'No exception recorded'
    step = 'run'
    try:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 16384))
            lines = stream.read().decode(errors='replace').splitlines()
        error = next((line.strip() for line in reversed(lines) if _EXCEPTION.match(line)), error)
        for marker in ('apply', 'verify'):
            if f'Version patch step: {marker}' in lines:
                step = marker
        if 'Patch verification did not return True' in error:
            step = 'verify'
    except OSError:
        pass

    if (error.startswith('Could not pause project interfaces:')
            or (attempt.get('error') or '').startswith('Could not pause project interfaces:')):
        step = 'suspend interfaces'
    # 0.7.3/001 has a published digest and used a generic lsof error.
    prefix = 'RuntimeError: Open files or uncertain lsof result: '
    if attempt['patch'] == '0.7.3/001' and error.startswith(prefix) and persist:
        try:
            check_open_files(Path(error.removeprefix(prefix)))
        except RuntimeError as exc:
            error = str(exc)

    blocked = []
    for path in sorted(backup.glob('*.json')):
        try:
            if path.stat().st_size <= 16 * 1024 * 1024:
                blocked.extend((path, row) for row in _blocked(json.loads(path.read_text())))
        except (OSError, ValueError, AttributeError):
            continue
    summary = dict(patch=attempt['patch'], step=step, error=error[:4000], blocked_count=len(blocked),
                   log_path=str(log), backup_dir=str(backup),
                   retry=f'.venv/bin/python -m src.automation.version_patch retry {attempt["patch"]}')
    if blocked:
        path, row = blocked[0]
        summary.update(report=str(path), step='apply', error=str(row.get('error') or row['reason'])[:4000])
        summary.update({key: row[key] for key in ('source', 'target', 'item') if key in row})
    if persist:
        atomic_json(destination, summary)
    return summary
