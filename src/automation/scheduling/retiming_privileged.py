"""Standalone, root-owned permission for calendar-only changes to existing timers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BASE = Path('/usr/local/lib/learndl-retiming')
SUDOERS = Path('/etc/sudoers.d')
UNITS = Path('/etc/systemd/system')


def envelope(content):
    return '\n'.join(line for line in content.splitlines() if not line.startswith('OnCalendar='))


def validate_calendar(content):
    calendars = [line.removeprefix('OnCalendar=') for line in content.splitlines() if line.startswith('OnCalendar=')]
    if not calendars or len(calendars) > 10080:
        raise ValueError('Expected a bounded weekly calendar')
    for value in calendars:
        match = re.fullmatch(r'(Sun|Mon|Tue|Wed|Thu|Fri|Sat) \*-\*-\* ([0-2][0-9]):([0-5][0-9]):00 ([A-Za-z0-9_+./-]+)', value)
        if not match or int(match[2]) > 23:
            raise ValueError('Invalid business calendar')


def execute(request, policy):
    argv = request['argv']
    allowed = policy['timers']
    # Recheck the paired service on every privileged operation.
    for name, record in allowed.items():
        service = UNITS / name.replace('.timer', '.service')
        if service.is_symlink() or hashlib.sha256(service.read_bytes()).hexdigest() != record['service_digest']:
            raise ValueError('Authorized service changed; authorize retiming again')
    if argv == ['sudo', 'systemctl', 'daemon-reload']:
        command = ['/usr/bin/systemctl', 'daemon-reload']
    elif (len(argv) in (4, 5) and argv[:2] == ['sudo', 'systemctl'] and argv[-1] in allowed
          and argv[2:-1] in [['disable', '--now'], ['enable', '--now'], ['enable'], ['disable'], ['start'], ['stop']]):
        command = ['/usr/bin/systemctl', *argv[2:]]
    elif len(argv) == 6 and argv[:4] == ['sudo', 'install', '-m', '0644']:
        name = Path(argv[-1]).name
        content = request.get('content', '')
        if name not in allowed or argv[-1] != str(UNITS / name):
            raise ValueError('Timer outside authorization')
        target = UNITS / name
        if target.is_symlink() or envelope(target.read_text()) != allowed[name]['envelope']:
            raise ValueError('Installed timer changed outside authorization')
        validate_calendar(content)
        if envelope(content) != allowed[name]['envelope']:
            raise ValueError('Only OnCalendar changes are authorized')
        zones = [line.rsplit(' ', 1)[-1] for line in content.splitlines() if line.startswith('OnCalendar=')]
        if any(zone != policy['timezone'] for zone in zones):
            raise ValueError('Timezone outside authorization')
        temporary = UNITS / ('.learndl-retiming-' + str(os.getpid()))
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return ''
    else:
        raise ValueError('Operation outside retiming authorization')
    return subprocess.run(command, check=True, text=True, capture_output=True, timeout=120).stdout


def setup(payload, source):
    uid = int(os.environ['SUDO_UID'])
    if uid == 0 or payload['uid'] != uid:
        raise ValueError('Invalid authorizing user')
    import pwd
    user = pwd.getpwuid(uid).pw_name
    directory = BASE / str(uid)
    for path in (BASE, directory):
        path.mkdir(mode=0o755, exist_ok=True)
        if path.is_symlink() or path.stat().st_uid != 0 or path.stat().st_mode & 0o022:
            raise ValueError('Unsafe permission directory')
    policy = directory / 'policy.json'
    helper = directory / 'helper.py'
    rule = SUDOERS / ('learndl-retiming-' + str(uid))
    if payload['action'] == 'revoke':
        policy.unlink(missing_ok=True)
        rule.unlink(missing_ok=True)
        return
    # Pin real installed service bytes, independently of caller-supplied hashes.
    for name, record in payload['timers'].items():
        if not re.fullmatch(r'learndl-schedule-[a-z][a-z0-9_]*\.timer', name):
            raise ValueError('Invalid business timer')
        timer, service = UNITS / name, UNITS / name.replace('.timer', '.service')
        if timer.is_symlink() or service.is_symlink():
            raise ValueError('Symlink units are unsupported')
        validate_calendar(timer.read_text())
        if envelope(timer.read_text()) != record['envelope']:
            raise ValueError('Timer changed during authorization')
        record['service_digest'] = hashlib.sha256(service.read_bytes()).hexdigest()
    for path, content in ((helper, source), (policy, json.dumps(payload))):
        if path.is_symlink():
            raise ValueError('Unsafe permission file')
        path.write_text(content)
        path.chmod(0o644)
    temporary = SUDOERS / ('.learndl-retiming-' + str(uid))
    temporary.write_text(f'{user} ALL=(root) NOPASSWD: /usr/bin/python3 -I {helper}\n')
    temporary.chmod(0o440)
    try:
        subprocess.run(['/usr/sbin/visudo', '-cf', str(temporary)], check=True, capture_output=True)
        temporary.replace(rule)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    uid = int(os.environ['SUDO_UID'])
    if os.getuid() != 0 or uid == 0:
        raise ValueError('Requires an authorized ordinary user')
    policy = json.loads((Path(__file__).parent / 'policy.json').read_text())
    if policy['uid'] != uid:
        raise ValueError('Wrong authorizing user')
    raw = sys.stdin.buffer.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Oversized request')
    sys.stdout.write(execute(json.loads(raw), policy))


if __name__ == '__main__':
    main()
