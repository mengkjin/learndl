"""Plan and install project schedules without taking ownership of external cron."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import grp
import json
import os
import pwd
import re
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import portalocker
import psutil

from .config import DAYS, ENTRYPOINTS, load_config, runtime_dir, schedule_slots
from src.automation.paths import ROOT, config_directory

BEGIN = '# BEGIN LEARNDL MANAGED'
END = '# END LEARNDL MANAGED'
PREFIX = 'learndl-schedule-'
UNIT_DIR = Path('/etc/systemd/system')


def desktop_path() -> Path:
    from src.proj import PATH
    name = hashlib.sha256(str(PATH.main.resolve()).encode()).hexdigest()[:12]
    return Path.home() / '.config' / 'autostart' / f'learndl-cli-{name}.desktop'


def desktop_entry(root: Path, python: Path) -> str:
    def quote(value: str) -> str:
        if any(c in value for c in '\n\r\0'):
            raise ValueError('Invalid desktop argument')
        for char in ('\\', '"', '`', '$'):
            value = value.replace(char, '\\' + char)
        return '"' + value.replace('%', '%%') + '"'
    argv = [str(python), str(root / 'src/automation/cli_recovery.py'), 'login',
            '--state', str(runtime_dir().parent / 'cli_recovery/state.json')]
    return ('[Desktop Entry]\nType=Application\nName=Learndl CLI recovery\n'
            'Comment=Refresh the desktop session for an opted-in Learndl menu\n'
            'Terminal=false\nExec=' + ' '.join(quote(arg) for arg in argv) + '\n')


def command(argv: list[str], *, input_text: str | None = None) -> str:
    if argv and argv[0] == 'sudo' and os.environ.get('LEARNDL_INSTALL_BROKER'):
        from .install_flow import broker_command
        return broker_command(argv)
    result = subprocess.run(argv, input=input_text, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {shlex.join(argv)}\n'
                           f'{result.stderr.strip()}\n{result.stdout.strip()}'.rstrip())
    return result.stdout


def read_cron() -> str:
    result = subprocess.run(['crontab', '-l'], text=True, capture_output=True)
    if result.returncode and 'no crontab for' not in result.stderr.lower():
        raise RuntimeError(result.stderr)
    return result.stdout


def timer_state(name: str) -> dict[str, str]:
    result = subprocess.run(['systemctl', 'show', name, '--property=ActiveState', '--property=UnitFileState'], text=True, capture_output=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def external_cron(text: str) -> str:
    if text.count(BEGIN) != text.count(END) or text.count(BEGIN) > 1:
        raise ValueError('Malformed managed crontab block')
    if BEGIN not in text:
        return text
    before, rest = text.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    return before + after.removeprefix('\n')


def field_values(value: str, maximum: int) -> set[int]:
    result = set()
    for term in value.split(','):
        base, *steps = term.split('/')
        step = int(steps[0]) if steps else 1
        if len(steps) > 1 or step <= 0:
            raise ValueError('Invalid cron step')
        if base == '*':
            low, high = 0, maximum
        elif '-' in base:
            low, high = map(int, base.split('-'))
        else:
            low = high = int(base)
        if not 0 <= low <= high <= maximum:
            raise ValueError('Invalid cron field')
        result.update(range(low, high + 1, step))
    return result


def cron_slots(fields: list[str]) -> set[tuple[int, int]]:
    minute, hour, dom, month, dow = fields
    if dom != '*' or month != '*':
        raise ValueError('Date-specific cron cannot be compared with weekly schedules')
    return {(day % 7, h * 60 + m) for day in field_values(dow, 7)
            for h in field_values(hour, 23) for m in field_values(minute, 59)}


def normalize_args(args: list[str]) -> list[str]:
    normalized = []
    for arg in args:
        normalized.extend(arg.split('=', 1) if arg.startswith('--') and '=' in arg else [arg])
    return normalized


def cron_coverage(text: str, config: dict, root: Path, local_timezone: str) -> tuple[dict, dict, list]:
    covered: dict[str, set] = {key: set() for key in config['tasks']}
    conflicts: dict[str, list[str]] = {}
    extras = []
    environment = False
    for line in external_cron(text).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if re.match(r'^[A-Za-z_][A-Za-z_0-9]*\s*=', stripped):
            # Environment can change executable resolution and semantics.
            # Conservatively refuse equivalence for affected project entries.
            environment = True
            continue
        candidates = [key for key, task in config['tasks'].items()
                      if f'runs/{task["entrypoint"]}.sh' in line or ENTRYPOINTS[task['entrypoint']] in line]
        if not candidates:
            if str(root) in line and ('task_monitor.watchdog' in line or 'automation.watchdog' in line or 'task_monitor_maintenance' in line or 'scheduling.idle_worklist' in line):
                extras.append('Existing watchdog cron requires explicit removal before installing its timer: ' + line)
            continue
        try:
            parts = stripped.split(None, 5)
            if len(parts) != 6:
                raise ValueError('Unsupported cron syntax')
            tokens = shlex.split(parts[5])
            if any(token in {'&&', '||', ';', '|', '&'} or any(c in token for c in ['>', '<', '`', '$', '\n']) for token in tokens):
                raise ValueError('Complex shell command')
            if tokens and tokens[0] in {'/bin/bash', '/usr/bin/bash', 'bash', '/bin/sh', 'sh'}:
                tokens = tokens[1:]
            matches = []
            for key in candidates:
                task = config['tasks'][key]
                path = root / 'runs' / f'{task["entrypoint"]}.sh'
                if tokens and tokens[0] == str(path) and normalize_args(tokens[1:]) == task['args']:
                    matches.append(key)
            if len(matches) != 1 or environment or config['timezone'] != local_timezone:
                raise ValueError('Ambiguous command, environment or timezone')
            key = matches[0]
            slots = cron_slots(parts[:5])
            if covered[key] & slots:
                raise ValueError('Duplicate active cron times')
            covered[key].update(slots)
        except (ValueError, IndexError) as exc:
            for key in candidates:
                conflicts.setdefault(key, []).append(f'{exc}: {line}')
    return covered, conflicts, extras


def unit_quote(value: str) -> str:
    if any(c in value for c in '\n\r\0'):
        raise ValueError('Invalid unit argument')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'


def service(root: Path, python: Path, module: str, args: list[str]) -> str:
    user = pwd.getpwuid(os.getuid())
    executable = ' '.join(unit_quote(str(v)) for v in [python, '-m', module, *args])
    return (f'[Unit]\nDescription=Learndl managed task\nAfter=network-online.target\n'
            f'[Service]\nType=oneshot\nUser={user.pw_name}\nWorkingDirectory={unit_path(root)}\n'
            f'Environment={unit_quote("HOME=" + user.pw_dir)}\nEnvironment={unit_quote("PYTHONPATH=" + str(root))}\n'
            f'ExecStart={executable}\nTimeoutStartSec=infinity\nTimeoutStopSec=30s\nKillMode=control-group\n'
            'NoNewPrivileges=true\nUMask=0077\n')


def protected_service(root, python, kind, args=()):
    from .autofix_store import state_dir
    value = service(root, python, 'placeholder', [])
    command_line = ' '.join(unit_quote(str(v)) for v in
                            [python, state_dir() / 'worker_bootstrap.py', kind, *args])
    return re.sub(r'^ExecStart=.*$', lambda _: 'ExecStart=' + command_line, value, flags=re.MULTILINE)


def unit_path(path: Path) -> str:
    """WorkingDirectory is a path directive, not a shell/ExecStart argument."""
    value = str(path)
    if not path.is_absolute() or any(c in value for c in '\n\r\0') or value.endswith('\\') or value != value.strip():
        raise ValueError('Unsupported working directory')
    return value.replace('%', '%%')


def next_times(slots: set, timezone: str) -> list[str]:
    current = datetime.now(ZoneInfo(timezone)).replace(second=0, microsecond=0) + timedelta(minutes=1)
    result = []
    for _ in range(7 * 1440):
        if ((current.weekday() + 1) % 7, current.hour * 60 + current.minute) in slots:
            result.append(current.isoformat())
            if len(result) == 3:
                break
        current += timedelta(minutes=1)
    return result


def build_plan(config: dict, cron: str, root: Path, python: Path, local_timezone: str) -> dict:
    config = dict(config, scheduler_version=3)
    coverage, conflicts, global_conflicts = cron_coverage(cron, config, root, local_timezone)
    desired: dict[str, set] = {key: set() for key in config['tasks']}
    for schedule in config['schedules'].values():
        desired[schedule['task']].update(schedule_slots(schedule))
    units = {}
    cron_lines = []
    report = {}
    for key, task in config['tasks'].items():
        wanted = desired[key] if task['enabled'] else set()
        missing = wanted - coverage[key]
        report[key] = {'cron_preserved': len(coverage[key]), 'cron_extra': len(coverage[key] - wanted),
                       'managed_slots': len(missing), 'backend': task['backend'],
                       'conflicts': conflicts.get(key, []), 'next': next_times(wanted, config['timezone']),
                       'timeout_seconds': task['timeout_seconds'], 'legacy_cron_timeout_protected': False}
        if key in conflicts:
            continue
        if task['backend'] == 'cron' and config['timezone'] != local_timezone:
            report[key]['conflicts'] = ['Cron backend requires server timezone to match configuration']
            conflicts[key] = report[key]['conflicts']
            continue
        if wanted:
            lane = 'learndl-lane-' + key
            units[lane + '.service'] = protected_service(root, python, 'lane', [key])
            units[lane + '.timer'] = ('[Unit]\nDescription=Learndl independent data lane\n[Timer]\n'
                'OnCalendar=*-*-* *:*:00\nAccuracySec=1s\nPersistent=false\n[Install]\nWantedBy=timers.target\n')
        name = PREFIX + key
        if missing and task['backend'] == 'systemd':
            units[name + '.service'] = protected_service(root, python, 'calendar', [key])
            calendars = '\n'.join(f'OnCalendar={DAYS[day].title()} *-*-* {minute // 60:02}:{minute % 60:02}:00 {config["timezone"]}' for day, minute in sorted(missing))
            units[name + '.timer'] = f'[Unit]\nDescription=Learndl schedule {key}\n[Timer]\n{calendars}\nAccuracySec=1s\nPersistent=false\n[Install]\nWantedBy=timers.target\n'
        elif missing:
            from .autofix_store import state_dir
            invocation = shlex.join([str(python), str(state_dir() / 'worker_bootstrap.py'), 'calendar', key])
            for day, minute in sorted(missing):
                cron_lines.append(f'{minute % 60} {minute // 60} * * {day} cd {shlex.quote(str(root))} && {invocation} >> {shlex.quote(str(runtime_dir() / "cron.log"))} 2>&1')
    if config['watchdog']['jobs'].get('autofix', {}).get('enabled'):
        units['learndl-autofix.service'] = protected_service(root, python, 'autofix') + 'RestrictAddressFamilies=AF_INET AF_INET6\n'
        units['learndl-autofix.timer'] = ('[Unit]\nDescription=Learndl autofix and deferred task worker\n[Timer]\n'
            'OnBootSec=90s\nOnUnitInactiveSec=60s\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n')
    # Installation admission also queues calendar slots when autofix is disabled.
    units['learndl-deferred.service'] = protected_service(root, python, 'deferred')
    units['learndl-deferred.timer'] = ('[Unit]\nDescription=Learndl deferred calendar tasks\n[Timer]\n'
        'OnBootSec=90s\nOnUnitInactiveSec=60s\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n')
    # Independent helper timers do not await SMTP/cache/desktop work in watchdog.
    from .auxiliary import NAMES
    for job in NAMES:
        if job != 'email_outbox' and not config['watchdog']['jobs'].get(job, {}).get('enabled'):
            continue
        name = 'learndl-aux-' + job.replace('_', '-')
        limit = max(60, config['watchdog']['jobs'].get(job, {}).get('max_seconds', 20) + 30) if job == 'task_monitor_cache' else 120
        units[name + '.service'] = protected_service(root, python, job).replace('TimeoutStartSec=infinity', f'TimeoutStartSec={limit}s')
        units[name + '.timer'] = ('[Unit]\nDescription=Learndl auxiliary worker\n[Timer]\n'
            'OnBootSec=90s\nOnUnitInactiveSec=60s\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n')
    units['learndl-git-update.service'] = service(root, python, 'src.automation.scheduling.git_update', []).replace('TimeoutStartSec=infinity', 'TimeoutStartSec=120s')
    units['learndl-git-update.timer'] = ('[Unit]\nDescription=Learndl Git check and maintenance\n[Timer]\n'
        'OnBootSec=60s\nOnUnitInactiveSec=60s\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n')
    # Watchdog always has one trigger, even when individual jobs are disabled.
    tick = config['watchdog']['tick_seconds']
    if config['default_backend'] == 'systemd':
        watchdog_limit = max(180, tick + 15 * len(config['watchdog']['units']) + config['watchdog']['jobs'].get('task_monitor_cache', {}).get('max_seconds', 20))
        units['learndl-watchdog.service'] = protected_service(root, python, 'watchdog').replace('TimeoutStartSec=infinity', f'TimeoutStartSec={watchdog_limit}s') + 'Nice=10\nOOMScoreAdjust=-900\nMemoryMax=1G\n'
        units['learndl-watchdog.timer'] = ('[Unit]\nDescription=Learndl maintenance\n[Timer]\n'
                                          f'OnBootSec=90s\nOnUnitInactiveSec={tick}s\nAccuracySec=1s\n'
                                          '[Install]\nWantedBy=timers.target\n')
    else:
        from .autofix_store import state_dir
        invocation = shlex.join([str(python), str(state_dir() / 'worker_bootstrap.py'), 'watchdog'])
        cron_lines.append(f'*/{tick // 60} * * * * cd {shlex.quote(str(root))} && {invocation} >> {shlex.quote(str(runtime_dir() / "cron.log"))} 2>&1')
    if any(config['watchdog']['jobs'].get(job, {}).get('enabled') for job in ('idle_worklist', 'rl_experiment')):
        if config['default_backend'] == 'systemd':
            units['learndl-idle-worklist.service'] = protected_service(root, python, 'training')
            units['learndl-idle-worklist.service'] += (
                'ExecStopPost=' + ' '.join(unit_quote(str(value)) for value in
                                          [python, '-m', 'src.automation.scheduling.recovery']) + '\n')
            units['learndl-idle-worklist.timer'] = ('[Unit]\nDescription=Learndl queued idle training\n[Timer]\n'
                'OnBootSec=90s\nOnUnitInactiveSec=60s\nAccuracySec=1s\n[Install]\nWantedBy=timers.target\n')
        else:
            from .autofix_store import state_dir
            invocation = shlex.join([str(python), str(state_dir() / 'worker_bootstrap.py'), 'training'])
            cron_lines.append(f'* * * * * cd {shlex.quote(str(root))} && {invocation} >> {shlex.quote(str(runtime_dir() / "cron.log"))} 2>&1')
    if any('%' in line for line in cron_lines):
        raise ValueError('Percent signs in cron paths are unsupported')
    new_cron = external_cron(cron).rstrip('\n') + '\n'
    if cron_lines:
        new_cron += BEGIN + '\n' + '\n'.join(cron_lines) + '\n' + END + '\n'
    desktop = desktop_entry(root, python) if config['watchdog']['jobs'].get('cli_recovery', {}).get('enabled') else None
    return {'config': config, 'units': units, 'cron': new_cron, 'report': report, 'desktop_autostart': desktop,
            'conflicts': conflicts, 'global_conflicts': global_conflicts,
            'input_hash': hashlib.sha256(cron.encode()).hexdigest()}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def validate_unit(name: str, contents: str) -> None:
    """Validate backups too: user-writable manifests cannot supply root units."""
    from src.proj import PATH
    allowed = {
        'Unit': {'Description', 'Documentation', 'After', 'Wants'},
        'Service': {'Type', 'User', 'Group', 'WorkingDirectory', 'Environment', 'ExecStart', 'ExecStopPost',
                    'TimeoutStartSec', 'TimeoutStopSec', 'KillMode', 'NoNewPrivileges', 'UMask',
                    'Nice', 'OOMScoreAdjust', 'MemoryMax', 'PrivateTmp', 'RestrictAddressFamilies'},
        'Timer': {'OnCalendar', 'OnBootSec', 'OnUnitInactiveSec', 'AccuracySec', 'Persistent', 'Unit'},
        'Install': {'WantedBy'},
    }
    section = ''
    fields: dict[str, list[str]] = {}
    for raw in contents.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1]
            if section not in allowed:
                raise ValueError('Unsupported unit section')
            continue
        key, separator, value = line.partition('=')
        if not separator or key not in allowed.get(section, set()) or line.endswith('\\'):
            raise ValueError(f'Unsupported unit directive: {line}')
        fields.setdefault(key, []).append(value)
    for key in ('After', 'Wants'):
        if key in fields and fields[key] != ['network-online.target']:
            raise ValueError('Unexpected service dependency')
    if 'WantedBy' in fields and fields['WantedBy'] != ['timers.target']:
        raise ValueError('Unexpected timer install target')
    if name.endswith('.service'):
        if fields.get('User') != [pwd.getpwuid(os.getuid()).pw_name] or fields.get('Type') != ['oneshot']:
            raise ValueError('Only ordinary-user oneshot services may be installed')
        if 'Group' in fields and fields['Group'] != [grp.getgrgid(os.getgid()).gr_name]:
            raise ValueError('Unexpected service group')
        if fields.get('NoNewPrivileges') not in (['true'], ['yes']):
            raise ValueError('NoNewPrivileges is required')
        if fields.get('WorkingDirectory') != [unit_path(PATH.main)]:
            raise ValueError('Unexpected working directory')
        starts = fields.get('ExecStart', [])
        if len(starts) != 1:
            raise ValueError('Exactly one fixed ExecStart is required')
        argv = shlex.split(starts[0])
        if not argv or Path(argv[0]).resolve() != Path(sys.executable).resolve():
            raise ValueError('Unexpected service interpreter')
        expected = ['-m', 'src.automation.watchdog'] if name == 'learndl-watchdog.service' else ['-m', 'src.automation.scheduling.runner', 'run', name.removeprefix(PREFIX).removesuffix('.service')]
        if name == 'learndl-autofix.service':
            if fields.get('RestrictAddressFamilies') != ['AF_INET AF_INET6']:
                raise ValueError('Autofix must not access host Unix control sockets')
            expected = ['-m', 'src.automation.scheduling.autofix', 'worker']
        if name == 'learndl-deferred.service':
            expected = ['-m', 'src.automation.scheduling.runner', 'deferred']
        if name == 'learndl-idle-worklist.service':
            expected = ['-m', 'src.automation.scheduling.idle_worklist', 'worker']
        from .autofix_store import state_dir
        kind = {'learndl-watchdog.service': 'watchdog', 'learndl-autofix.service': 'autofix', 'learndl-deferred.service': 'deferred',
                'learndl-idle-worklist.service': 'training'}.get(name)
        arguments = []
        if name.startswith(PREFIX):
            kind, arguments = 'calendar', [name.removeprefix(PREFIX).removesuffix('.service')]
        if name.startswith('learndl-aux-'):
            kind = name.removeprefix('learndl-aux-').removesuffix('.service').replace('-', '_')
            if kind not in {'email_outbox', 'cli_recovery', 'task_monitor_cache'}:
                raise ValueError('Unexpected auxiliary worker')
            expected = None
        if name.startswith('learndl-lane-'):
            kind, arguments = 'lane', [name.removeprefix('learndl-lane-').removesuffix('.service')]
            expected = None
        if name == 'learndl-git-update.service':
            expected = ['-m', 'src.automation.scheduling.git_update']
        protected = [str(state_dir() / 'worker_bootstrap.py'), kind, *arguments] if kind else None
        # Old snapshots remain valid for upgrades and rollback. Do not normalize
        # arbitrary commands: only the exact historical module spelling is allowed.
        legacy = [part.replace('src.automation.', 'src.api.task_monitor.') for part in expected] if expected else None
        if argv[1:] != expected and argv[1:] != legacy and argv[1:] != protected:
            raise ValueError('Unexpected service command')
        if 'ExecStopPost' in fields:
            if name != 'learndl-idle-worklist.service' or len(fields['ExecStopPost']) != 1:
                raise ValueError('Unexpected stop observer')
            observer = shlex.split(fields['ExecStopPost'][0])
            if (len(observer) != 3 or Path(observer[0]).resolve() != Path(sys.executable).resolve()
                    or observer[1:] not in (['-m', 'src.automation.scheduling.recovery'],
                                           ['-m', 'src.api.task_monitor.scheduling.recovery'])):
                raise ValueError('Unexpected stop observer command')
        for values in fields.get('Environment', []):
            for assignment in shlex.split(values):
                key, _, value = assignment.partition('=')
                if key == 'HOME' and value == pwd.getpwuid(os.getuid()).pw_dir:
                    continue
                if key == 'PYTHONPATH' and value == str(PATH.main):
                    continue
                if key == 'LEARNDL_WATCHDOG_UNITS' and re.fullmatch(r'[A-Za-z0-9_.@:, -]*', value):
                    continue
                raise ValueError('Unsupported service environment')
    elif fields.get('Unit', [name.removesuffix('.timer') + '.service']) != [name.removesuffix('.timer') + '.service']:
        raise ValueError('Timer must target its matching project service')


def apply_plan(plan: dict, original_cron: str, directory: Path, *, installation_id=None) -> None:
    # Serialize installation/rollback against dispatch and the full task lifetime,
    # not merely a point-in-time "no active rows" check.
    from .autofix_store import Store, state_dir
    control = state_dir()
    if control.exists():
        from .maintenance_gate import permit
        Store(control)
        with portalocker.Lock(str(control / 'worker.lock'), mode='r+', timeout=0), \
             permit(directory=control, exclusive=True, blocking=False, installation_id=installation_id):
            return _apply_plan(plan, original_cron, directory)
    return _apply_plan(plan, original_cron, directory)


def _apply_plan(plan: dict, original_cron: str, directory: Path) -> None:
    if plan['global_conflicts']:
        raise ValueError('\n'.join(plan['global_conflicts']))
    manifest_path = directory / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'units': {}}
    transaction_path = directory / 'transaction.json'
    transaction = json.loads(transaction_path.read_text()) if transaction_path.exists() else {}
    recovering = transaction.get('phase') not in {None, 'complete'}
    desktop = desktop_path()
    actual_desktop = desktop.read_text() if desktop.exists() else None
    desired_desktop = plan.get('desktop_autostart')
    interrupted_desktop = transaction.get('plan', {}).get('desktop_autostart') if recovering else None
    if actual_desktop is not None and actual_desktop not in (previous.get('desktop_autostart'), desired_desktop, interrupted_desktop):
        raise RuntimeError(f'Desktop autostart was externally changed: {desktop}')
    for name in previous['units']:
        if not re.fullmatch(r'(learndl-(?:schedule|lane)-[a-z][a-z0-9_]*|learndl-git-update|learndl-watchdog|learndl-idle-worklist|learndl-autofix|learndl-deferred|learndl-aux-(?:email-outbox|cli-recovery|task-monitor-cache))\.(timer|service)', name):
            raise ValueError('Invalid manifest unit name')
    # Do not convert a still-running legacy script whose whole-shell lifetime
    # is not covered by the new runner lock (daily has post-main schedulers).
    if plan['config']:
        from .runtime import RunStore
        owned = {run.get('pid') for run in RunStore(directory / 'runs.sqlite').active()}
        for process in psutil.process_iter(['pid', 'cmdline']):
            if process.pid in owned:
                continue
            argv = process.info.get('cmdline') or []
            for key, task in plan['config']['tasks'].items():
                expected = str(ROOT / ENTRYPOINTS[task['entrypoint']])
                if expected in argv and plan['report'].get(key, {}).get('managed_slots'):
                    plan['conflicts'].setdefault(key, []).append(f'Legacy PID {process.pid} is still running; installation deferred')
    # Adopt the pre-existing project watchdog, including its exact contents for rollback.
    for name in ('learndl-watchdog.service', 'learndl-watchdog.timer'):
        path = UNIT_DIR / name
        if name not in previous['units'] and path.exists():
            previous['units'][name] = path.read_text()
    if recovering:
        # Include partially installed files when deriving the recovery diff.
        for name in transaction['plan']['units']:
            if not re.fullmatch(r'(learndl-(?:schedule|lane)-[a-z][a-z0-9_]*|learndl-git-update|learndl-watchdog|learndl-idle-worklist|learndl-autofix|learndl-deferred|learndl-aux-(?:email-outbox|cli-recovery|task-monitor-cache))\.(timer|service)', name):
                raise ValueError('Invalid transaction unit name')
            path = UNIT_DIR / name
            if path.exists():
                previous['units'][name] = path.read_text()
    # Preserve previously installed configuration for tasks blocked by ambiguity.
    old_config_path = directory / 'installed.json'
    old_config = json.loads(old_config_path.read_text()) if old_config_path.exists() else None
    if old_config and old_config.get('scheduler_version', 1) < 3 and plan['config'] and plan['config'].get('scheduler_version', 1) >= 3 and plan['conflicts']:
        raise RuntimeError('Resolve legacy schedule conflicts before upgrading channel workers')
    if old_config and old_config.get('autofix_initialized') and plan['config']:
        plan['config']['autofix_initialized'] = True
    if old_config and old_config.get('autofix_initialized'):
        from .autofix_store import Store, state_dir
        controls = Store(state_dir())
        if any(r['phase'] in {'starting', 'running'} for r in controls.rows()):
            raise RuntimeError('Wait for the running autofix before changing installation')
        enabled = bool(plan['config'] and plan['config']['watchdog']['jobs'].get('autofix', {}).get('enabled'))
        if not enabled and controls.scheduled() and 'learndl-deferred.service' not in plan['units']:
            raise RuntimeError('Drain deferred calendar tasks before disabling autofix')
    for key in plan['conflicts']:
        if BEGIN in plan['cron']:
            before, managed = plan['cron'].split(BEGIN, 1)
            block, after = managed.split(END, 1)
            kept = [line for line in block.splitlines() if f'run {key} ' not in line and f'calendar {key} ' not in line]
            plan['cron'] = before + BEGIN + '\n' + '\n'.join(line for line in kept if line) + '\n' + END + after
        for name in list(plan['units']):
            if name.startswith((PREFIX + key + '.', 'learndl-lane-' + key + '.')):
                del plan['units'][name]
        for name, contents in previous['units'].items():
            if name.startswith((PREFIX + key + '.', 'learndl-lane-' + key + '.')):
                plan['units'][name] = contents
        if old_config and key in old_config['tasks']:
            plan['config']['tasks'][key] = old_config['tasks'][key]
    if plan['conflicts'] and BEGIN in original_cron:
        block = original_cron.split(BEGIN, 1)[1].split(END, 1)[0]
        preserved = [line for line in block.splitlines() if any(f'run {key} ' in line or f'calendar {key} ' in line for key in plan['conflicts'])]
        if preserved:
            if BEGIN not in plan['cron']:
                plan['cron'] += BEGIN + '\n' + END + '\n'
            plan['cron'] = plan['cron'].replace(END, '\n'.join(preserved) + '\n' + END)
    for key in plan['conflicts']:
        if not old_config or key not in old_config['tasks']:
            plan['config']['tasks'][key]['enabled'] = False
    snapshot = {'cron': original_cron, 'manifest': previous, 'config': old_config,
                'desktop_autostart': actual_desktop}
    previous['timer_states'] = {name: timer_state(name) for name in previous['units'] if name.endswith('.timer')}
    files_match = all((UNIT_DIR / name).exists() and (UNIT_DIR / name).read_text() == contents for name, contents in plan['units'].items())
    timers_match = all(timer_state(name).get('ActiveState') == 'active' and timer_state(name).get('UnitFileState') == 'enabled' for name in plan['units'] if name.endswith('.timer'))
    policy_matches = True
    if plan['config'] and plan['config']['watchdog']['jobs'].get('autofix', {}).get('enabled'):
        from .autofix_store import state_dir, Store
        control = state_dir(ROOT)
        if old_config and old_config.get('autofix_initialized'):
            Store(control)
        policy_matches = (control / 'policy.py').exists() and (control / 'policy.py').read_bytes() == Path(__file__).with_name('autofix_sandbox.py').read_bytes()
    if plan['config'] and plan['config'].get('scheduler_version', 1) >= 2:
        from .autofix_store import state_dir
        bootstrap = state_dir() / 'worker_bootstrap.py'
        policy_matches = policy_matches and bootstrap.exists() and bootstrap.read_bytes() == Path(__file__).with_name('worker_bootstrap.py').read_bytes()
        settings = state_dir() / 'worker-settings.json'
        policy_matches = policy_matches and settings.exists() and json.loads(settings.read_text()) == {'root': str(ROOT), 'runtime': str(directory)}
        if old_config and old_config.get('scheduler_version', 1) >= 2:
            from .priority import read
            from .autofix_store import Store
            read(Store(state_dir()))  # Missing authority is never silently reset.
            if old_config.get('scheduler_version', 1) >= 3:
                from .channels import read as read_channels
                read_channels(Store(state_dir()))
    if not recovering and policy_matches and files_match and timers_match and actual_desktop == desired_desktop and (plan['units'] == previous['units'] and plan['cron'] == original_cron and plan['config'] == old_config):
        print('Already up to date')
        return
    # Explicit allow-list: never manipulate an unrelated unit through a manifest.
    names = set(previous['units']) | set(plan['units'])
    if any(not re.fullmatch(r'(learndl-(?:schedule|lane)-[a-z][a-z0-9_]*|learndl-git-update|learndl-watchdog|learndl-idle-worklist|learndl-autofix|learndl-deferred|learndl-aux-(?:email-outbox|cli-recovery|task-monitor-cache))\.(timer|service)', name) for name in names):
        raise ValueError('Invalid managed unit name')
    with tempfile.TemporaryDirectory(prefix='learndl-install-') as temp:
        stage = Path(temp)
        for name, contents in plan['units'].items():
            validate_unit(name, contents)
            (stage / name).write_text(contents)
        if plan['units']:
            command(['systemd-analyze', 'verify', *[str(stage / name) for name in plan['units']]])
        if read_cron() != original_cron:
            raise RuntimeError('Crontab changed during planning; rerun installer')
        if plan['config'] and (plan['config'].get('scheduler_version', 1) >= 2 or plan['config']['watchdog']['jobs'].get('autofix', {}).get('enabled')):
            from .autofix_sandbox import preflight
            from .autofix_store import initialize, state_dir
            if plan['config']['watchdog']['jobs'].get('autofix', {}).get('enabled'):
                preflight()
            control = state_dir(ROOT)
            if old_config and old_config.get('autofix_initialized') and not control.exists():
                raise RuntimeError('Autofix ledger disappeared; restore it instead of reinitializing')
            initialize(control, ROOT)
            if plan['config'].get('scheduler_version', 1) >= 2:
                from .priority import initialize as initialize_priority
                from .autofix_store import Store
                initialize_priority(Store(control))
                from .channels import initialize as initialize_channels
                initialize_channels(Store(control))
                bootstrap = Path(__file__).with_name('worker_bootstrap.py')
                temporary_bootstrap = control / 'worker_bootstrap.tmp'
                temporary_bootstrap.write_bytes(bootstrap.read_bytes())
                temporary_bootstrap.replace(control / 'worker_bootstrap.py')
                atomic_json(control / 'worker-settings.json', {'root': str(ROOT), 'runtime': str(directory)})
            # Policy is installed explicitly, never replaced by the sandboxed script.
            source = Path(__file__).with_name('autofix_sandbox.py')
            policy = control / 'policy.py'
            temporary = control / 'policy.tmp'
            temporary.write_bytes(source.read_bytes())
            temporary.replace(policy)
            plan['config']['autofix_initialized'] = True
        if not recovering:
            atomic_json(directory / 'rollback.json', snapshot)
        atomic_json(directory / 'transaction.json', {'phase': 'prepared', 'plan': plan})
        changed = {name for name in names if previous['units'].get(name) != plan['units'].get(name)}
        changed.update(name for name, contents in plan['units'].items() if not (UNIT_DIR / name).exists() or (UNIT_DIR / name).read_text() != contents)
        for name in sorted(changed):
            if name.endswith('.timer') and (UNIT_DIR / name).exists():
                command(['sudo', 'systemctl', 'disable', '--now', name])
        atomic_json(directory / 'transaction.json', {'phase': 'paused', 'plan': plan})
        for name in sorted(names - set(plan['units'])):
            path = UNIT_DIR / name
            if path.exists():
                command(['sudo', 'mv', str(path), str(path) + '.learndl-disabled'])
        for name in sorted(changed & set(plan['units'])):
            command(['sudo', 'install', '-m', '0644', str(stage / name), str(UNIT_DIR / name)])
        # Snapshot publication precedes activation; running tasks keep their own policy.
        if desired_desktop is None:
            desktop.unlink(missing_ok=True)
        else:
            desktop.parent.mkdir(parents=True, exist_ok=True)
            temporary = desktop.with_suffix('.tmp')
            temporary.write_text(desired_desktop)
            temporary.chmod(0o600)
            temporary.replace(desktop)
        if plan['config'] is None:
            old_config_path.unlink(missing_ok=True)
        else:
            atomic_json(old_config_path, plan['config'])
        from .autofix_store import Store, state_dir
        if state_dir().exists():
            Store(state_dir()).set_enabled(bool(plan['config'] and plan['config'].get('autofix_initialized')
                and plan['config']['watchdog']['jobs'].get('autofix', {}).get('enabled')))
        if read_cron() != original_cron:
            raise RuntimeError('Crontab changed during installation; transaction preserved for rollback')
        if plan['cron'] != original_cron:
            command(['crontab', '-'], input_text=plan['cron'])
        if names:
            command(['sudo', 'systemctl', 'daemon-reload'])
        for name in sorted(plan['units']):
            if name.endswith('.timer'):
                restore_state = plan.get('timer_states', {}).get(name)
                if restore_state is None:
                    command(['sudo', 'systemctl', 'enable', '--now', name])
                else:
                    command(['sudo', 'systemctl', 'enable' if restore_state.get('UnitFileState') == 'enabled' else 'disable', name])
                    command(['sudo', 'systemctl', 'start' if restore_state.get('ActiveState') == 'active' else 'stop', name])
        atomic_json(manifest_path, {'units': plan['units'], 'cron': plan['cron'], 'desktop_autostart': desired_desktop})
        atomic_json(directory / 'transaction.json', {'phase': 'complete'})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'apply', 'defer', 'pending', 'cancel', 'status', 'rollback'])
    parser.add_argument('--host', default=socket.gethostname().split('.')[0])
    parser.add_argument('--config-dir', type=Path)
    parser.add_argument('--crontab-file', type=Path, help='Read-only offline planning input')
    parser.add_argument('--server-timezone', help='Required for offline planning; apply detects actual timezone')
    parser.add_argument('--diff', action='store_true', help='Show exact cron/unit changes in plan output')
    args = parser.parse_args()
    if args.action in {'apply', 'defer', 'pending', 'cancel'}:
        from .install_flow import run_interactive
        from . import install_request
        if args.crontab_file or args.server_timezone:
            parser.error('Installation uses live server configuration')
        if args.action in {'pending', 'cancel'}:
            print(json.dumps(install_request.cancel() if args.action == 'cancel' else install_request.reconcile(), indent=2))
            return 0
        return run_interactive(deferred=args.action == 'defer', host=args.host, config_dir=args.config_dir)
    from src.proj import PATH
    root = Path(PATH.main)
    if args.crontab_file and args.action != 'plan':
        parser.error('--crontab-file is only supported for plan')
    if args.action == 'apply' and args.server_timezone:
        parser.error('apply detects server timezone; do not override it')
    cron = args.crontab_file.read_text() if args.crontab_file else read_cron()
    if args.action != 'rollback':
        config = load_config(config_directory(args.config_dir, root=root), args.host)
        timezone = args.server_timezone or command(['timedatectl', 'show', '--property=Timezone', '--value']).strip()
        plan = build_plan(config, cron, root, Path(sys.executable), timezone)
    else:
        plan = {}
    if args.action in {'plan', 'status'}:
        print(json.dumps({key: value for key, value in plan.items() if key not in {'config', 'units', 'cron', 'desktop_autostart'}}, indent=2))
        desktop = desktop_path()
        actual_desktop = desktop.read_text() if desktop.exists() else None
        print(json.dumps({'desktop_autostart': str(desktop),
                          'desktop_autostart_changes': actual_desktop != plan.get('desktop_autostart')}))
        manifest_path = runtime_dir() / 'manifest.json'
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'units': {}}
        print(json.dumps({'add_units': sorted(set(plan['units']) - set(previous['units'])),
                          'remove_units': sorted(set(previous['units']) - set(plan['units'])),
                          'update_units': sorted(name for name in set(plan['units']) & set(previous['units']) if plan['units'][name] != previous['units'][name]),
                          'cron_changes': plan['cron'] != cron}, indent=2))
        if args.diff:
            print(''.join(difflib.unified_diff((actual_desktop or '').splitlines(True),
                      (plan.get('desktop_autostart') or '').splitlines(True), fromfile=str(desktop), tofile='planned autostart')))
            print(''.join(difflib.unified_diff(cron.splitlines(True), plan['cron'].splitlines(True), fromfile='current crontab', tofile='planned crontab')))
            for name in sorted(set(previous['units']) | set(plan['units'])):
                print(''.join(difflib.unified_diff(previous['units'].get(name, '').splitlines(True), plan['units'].get(name, '').splitlines(True), fromfile='current/' + name, tofile='planned/' + name)))
        if args.action == 'status':
            from .priority import status as priority_status
            print(json.dumps({'scheduling_priority': priority_status()}, indent=2, ensure_ascii=False))
            from src.automation.cli_recovery import load, state_path
            print(json.dumps({'cli_recovery': load(state_path())}, indent=2))
            directory = runtime_dir()
            manifest = json.loads((directory / 'manifest.json').read_text()) if (directory / 'manifest.json').exists() else {}
            print(json.dumps({'installed': bool(manifest), 'cron_matches': manifest.get('cron') == cron,
                              'unit_files_match': all((UNIT_DIR / name).exists() and (UNIT_DIR / name).read_text() == contents for name, contents in manifest.get('units', {}).items()),
                              'timers': {name: timer_state(name) for name in manifest.get('units', {}) if name.endswith('.timer')}}))
        return int(bool(plan['conflicts'] or plan['global_conflicts']))
    if sys.platform != 'linux' or os.getuid() == 0:
        raise RuntimeError('Run installation as the ordinary service user on Linux; sudo is used only for unit installation')
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(directory / 'install.lock'), timeout=1):
        if args.action == 'rollback':
            backup = json.loads((directory / 'rollback.json').read_text())
            manifest = json.loads((directory / 'manifest.json').read_text()) if (directory / 'manifest.json').exists() else {}
            transaction = json.loads((directory / 'transaction.json').read_text()) if (directory / 'transaction.json').exists() else {}
            expected = transaction.get('plan', {}).get('cron', manifest.get('cron'))
            if cron not in {backup['cron'], expected}:
                raise RuntimeError('External cron changed; inspect and resolve before rollback')
            plan = {'config': backup['config'], 'units': backup['manifest']['units'], 'cron': backup['cron'], 'conflicts': {}, 'global_conflicts': [], 'report': {},
                    'desktop_autostart': backup.get('desktop_autostart'),
                    'timer_states': backup['manifest'].get('timer_states', {})}
        from . import install_request
        pending_installation = install_request.reconcile()
        repair_id = pending_installation.get('id') if pending_installation.get('phase') == 'repair_required' else None
        apply_plan(plan, cron, directory, installation_id=repair_id)
        if repair_id:
            with install_request.lock():
                install_request.finish(pending_installation, 'failed', 'Installation rolled back explicitly; task admission restored')
    if plan['conflicts']:
        print(json.dumps({'deferred_tasks': plan['conflicts']}, indent=2))
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'Scheduling installer: {exc}', file=sys.stderr)
        raise SystemExit(1) from None
