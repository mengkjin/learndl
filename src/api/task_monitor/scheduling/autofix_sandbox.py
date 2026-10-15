"""Linux-only filesystem isolation. No unsandboxed fallback is supported."""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

READONLY = ('.venv', '.git', '.secret')


def check_mounts(root: Path, mountinfo: str):
    # ismount() alone misses bind mounts on the same filesystem.
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 6 or ' - ' not in line:
            raise ValueError('Cannot verify mount topology')
        mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), fields[4]))
        if mount != root and mount.is_relative_to(root):
            raise ValueError(f'Nested mount in writable project: {mount}')


def preflight() -> str:
    if sys.platform != 'linux':
        raise RuntimeError('Autofix execution requires Linux and Bubblewrap; preview works on other platforms')
    executable = shutil.which('bwrap')
    if not executable:
        raise RuntimeError('Bubblewrap (bwrap) is not installed; refusing unsandboxed execution')
    result = subprocess.run([executable, '--help'], capture_output=True, text=True, timeout=10, check=True)
    help_text = result.stdout + result.stderr
    if '--disable-userns' not in help_text:
        version = 'version unavailable'
        try:
            result = subprocess.run([executable, '--version'], capture_output=True, text=True, timeout=10)
            version = (result.stdout or result.stderr).strip()[:200] or version
        except (OSError, subprocess.SubprocessError):
            pass
        raise RuntimeError(
            f'Bubblewrap at {executable} ({version}) lacks --disable-userns. '
            'This capability was added in upstream 0.8.0; install a supported distribution package '
            'providing it, verify the bwrap selected by PATH, then rerun scheduling plan/apply. '
            'Autofix isolation has not been downgraded. See runs/autofix/README.md.')
    return executable


def check_tree(root: Path):
    """Reject nested mounts, special files, and writable aliases outside this tree."""
    if root != root.resolve():
        raise ValueError('Writable project root must be canonical')
    if sys.platform == 'linux':
        check_mounts(root, Path('/proc/self/mountinfo').read_text())
    links = {}
    def fail(error):
        raise error  # Never silently skip an unreadable portion of the tree.
    for base, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        base = Path(base)
        if base == root:
            dirs[:] = [name for name in dirs if name not in READONLY]
        for name in list(dirs) + files:
            path = base / name
            meta = path.lstat()
            if path.is_symlink():
                continue  # Symlink targets outside the RW mount remain read-only.
            if os.path.ismount(path):
                raise ValueError(f'Nested mount in writable project: {path}')
            if stat.S_ISREG(meta.st_mode) and meta.st_nlink > 1:
                key = (meta.st_dev, meta.st_ino)
                count, expected = links.get(key, (0, meta.st_nlink))
                links[key] = (count + 1, expected)
            elif not (stat.S_ISDIR(meta.st_mode) or stat.S_ISREG(meta.st_mode)):
                raise ValueError(f'Special file/socket in writable project: {path}')
    if any(count != expected for count, expected in links.values()):
        raise ValueError('Project contains a hard link to a file outside the writable tree')


def command(root: Path, request: Path, argv: list[str], *, executable=None) -> list[str]:
    executable = executable or preflight()
    root = root.resolve()
    # Empty private /run removes host D-Bus/control sockets; /proc is private.
    args = [executable, '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts',
            '--disable-userns', '--new-session', '--die-with-parent', '--cap-drop', 'ALL',
            '--ro-bind', '/', '/',
            '--tmpfs', '/tmp', '--tmpfs', '/var/tmp', '--tmpfs', '/run', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/dev/shm',
            '--bind', str(root), str(root)]
    for name in READONLY:
        path = root / name
        if path.exists():
            args += ['--ro-bind', str(path.resolve()), str(path)]
    # A read-only immutable context, not an authority writable by the script.
    args += ['--ro-bind', str(request), '/run/learndl-autofix.json', '--chdir', str(root),
             '--setenv', 'TMPDIR', '/tmp', '--setenv', 'XDG_CACHE_HOME', '/tmp/cache',
             '--unsetenv', 'DBUS_SESSION_BUS_ADDRESS', '--unsetenv', 'SSH_AUTH_SOCK',
             '--unsetenv', 'LEARNDL_MANAGED_RUN', '--unsetenv', 'LEARNDL_MAINTENANCE_FD', '--', *argv]
    return args


def script_event(kind: str, **fields):
    """Sandbox -> supervisor mailbox. This never opens the authoritative ledger."""
    from .maintenance_gate import sandbox_request
    request = sandbox_request()
    if request is None:
        return
    # Root AutoRunTask only. Nested operations keep their own normal behaviour.
    payload = {'id': request['id'], 'nonce': request['nonce'], 'kind': kind, **fields}
    folder = Path(request['channel'])
    path = folder / f'{kind}.json'
    temporary = folder / f'.{kind}.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(payload))
    temporary.replace(path)
