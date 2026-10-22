"""Standalone stdlib-only one-shot privilege broker, frozen under /run at authorization.

Never import project code as root. The worker runs as the authorizing user. Requests
can only install approved bytes or operate on the exact approved service names.
"""
from __future__ import annotations

import json
import os
import pwd
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

BASE = Path('/run/learndl-install')
UNITS = Path('/etc/systemd/system')
NAME = re.compile(r'learndl-(?:schedule-[a-z][a-z0-9_]*|watchdog|idle-worklist|autofix|deferred)\.(?:service|timer)')


def execute(request: dict, approval: dict) -> str:
    if time.time() > approval['expires_at'] + 3600:
        raise ValueError('Installation privilege authorization expired')
    argv = request['argv']
    allowed = approval['allowed_units']
    if argv == ['sudo', 'systemctl', 'daemon-reload']:
        command = ['/usr/bin/systemctl', 'daemon-reload']
    elif (len(argv) in (4, 5) and argv[:2] == ['sudo', 'systemctl'] and argv[-1] in allowed
          and argv[-1].endswith('.timer') and argv[2:-1] in
          [['disable', '--now'], ['enable', '--now'], ['enable'], ['disable'], ['start'], ['stop']]):
        command = ['/usr/bin/systemctl', *argv[2:]]
    elif len(argv) == 6 and argv[:4] == ['sudo', 'install', '-m', '0644']:
        name = Path(argv[-1]).name
        content = request.get('content')
        if not isinstance(content, str) or argv[-1] != str(UNITS / name) or name not in allowed or content not in allowed[name]:
            raise ValueError('Unapproved unit installation')
        target = UNITS / name
        if target.is_symlink():
            raise ValueError('Refusing symlink unit destination')
        temporary = UNITS / ('.learndl-install-' + approval['id'])
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
        temporary.replace(target)
        return ''
    elif len(argv) == 4 and argv[:2] == ['sudo', 'mv']:
        name = Path(argv[2]).name
        if name not in allowed or argv[2:] != [str(UNITS / name), str(UNITS / name) + '.learndl-disabled']:
            raise ValueError('Unapproved unit removal')
        if (UNITS / name).is_symlink() or Path(argv[3]).is_symlink():
            raise ValueError('Refusing symlink unit destination')
        (UNITS / name).replace(argv[3])
        return ''
    else:
        raise ValueError('Unapproved privileged command')
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def serve(bundle: Path) -> int:
    approval = json.loads((bundle / 'approval.json').read_text())
    user = pwd.getpwuid(approval['uid'])
    address = bundle / 'broker.sock'
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(address))
        os.chown(address, user.pw_uid, user.pw_gid)
        os.chmod(address, 0o600)
        server.listen(4)
        server.settimeout(1)
        env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': user.pw_dir, 'USER': user.pw_name,
               'LOGNAME': user.pw_name, 'PYTHONUNBUFFERED': '1', 'LEARNDL_INSTALL_BROKER': str(address)}
        child = subprocess.Popen([approval['python'], '-m',
            'src.api.task_monitor.scheduling.install_flow', 'worker', str(bundle)],
            cwd=approval['root'], env=env, user=user.pw_uid, group=user.pw_gid,
            extra_groups=os.getgrouplist(user.pw_name, user.pw_gid), stdin=subprocess.DEVNULL)
        (bundle / 'ready').write_text(str(child.pid))
        try:
            while child.poll() is None:
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(10)
                    _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, getattr(socket, 'SO_PEERCRED'), 12))
                    try:
                        if uid != user.pw_uid:
                            raise ValueError('Wrong broker peer')
                        data = connection.makefile('rb').readline(2_000_001)
                        if len(data) > 2_000_000:
                            raise ValueError('Oversized request')
                        result = {'output': execute(json.loads(data), approval)}
                    except Exception as exc:
                        result = {'error': str(exc)}
                    connection.sendall((json.dumps(result) + '\n').encode())
        finally:
            (bundle / 'result.json').write_text(json.dumps({'returncode': child.poll(), 'at': time.time()}))
    return child.wait()


def register(payload: Path) -> None:
    approval = json.loads(payload.read_text())
    uid = int(os.environ['SUDO_UID'])
    if uid == 0 or approval['uid'] != uid or not re.fullmatch('[a-f0-9]{32}', approval['id']):
        raise ValueError('Invalid authorizing identity')
    if approval['unit'] != f'learndl-install-{approval["id"]}.service':
        raise ValueError('Invalid installation service name')
    if any(not NAME.fullmatch(name) for name in approval['allowed_units']):
        raise ValueError('Invalid unit name')
    BASE.mkdir(mode=0o755, exist_ok=True)
    if BASE.is_symlink() or BASE.stat().st_uid != 0 or BASE.stat().st_mode & 0o022:
        raise ValueError('Unsafe installation staging directory')
    os.chmod(BASE, 0o755)
    bundle = BASE / approval['id']
    bundle.mkdir(mode=0o755)
    os.chmod(bundle, 0o755)
    (bundle / 'approval.json').write_text(json.dumps(approval))
    source = globals().get('__source__')
    (bundle / 'broker.py').write_text(source if source is not None else Path(__file__).read_text())
    os.chmod(bundle / 'approval.json', 0o644)
    os.chmod(bundle / 'broker.py', 0o644)
    result = subprocess.run(['/usr/bin/systemd-run', '--quiet', '--collect',
        '--unit=' + approval['unit'], '--property=Type=exec', '--property=KillMode=control-group',
        '--property=RuntimeMaxSec=90000', '/usr/bin/python3', '-I', str(bundle / 'broker.py'),
        'serve', str(bundle)], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)


if __name__ == '__main__':
    if os.getuid() != 0:
        raise SystemExit('This helper requires explicit administrator authorization')
    if sys.argv[1] == 'register':
        register(Path(sys.argv[2]))
    elif sys.argv[1] == 'serve':
        raise SystemExit(serve(Path(sys.argv[2])))
    else:
        raise SystemExit('Unknown helper action')
