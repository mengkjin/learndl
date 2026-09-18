"""Recover one explicitly registered Linux desktop CLI; never replay research work.

The desktop autostart entry refreshes session credentials at login. Watchdog
dispatches through the user service manager so its own service exit cannot kill
the recovered window. This file can run directly without importing the project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import portalocker
import psutil

ENV_KEYS = ('DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR',
            'DBUS_SESSION_BUS_ADDRESS', 'XDG_SESSION_ID', 'WEZTERM_UNIX_SOCKET', 'PATH')


def state_path() -> Path:
    from src.proj import PATH
    return PATH.runtime / 'cli_recovery' / 'state.json'


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def save(path: Path, state: dict) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)


def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return portalocker.Lock(str(path.with_suffix('.lock')), timeout=1)


def process_identity() -> dict:
    return {'pid': os.getpid(), 'created': psutil.Process().create_time()}


def live(process: dict) -> bool:
    try:
        item = psutil.Process(process['pid'])
        return item.create_time() == process['created'] and item.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError):
        return False


def desktop_environment() -> dict:
    return {key: os.environ[key] for key in ENV_KEYS if os.environ.get(key)}


def terminal_identity() -> dict:
    """Remember the GUI ancestor when available, to detect a stranded live Hub."""
    try:
        for process in psutil.Process().parents():
            if 'wezterm' in process.name().lower():
                return {'pid': process.pid, 'created': process.create_time()}
    except (psutil.Error, OSError):
        pass
    return {}


def hub_alive(state: dict) -> bool:
    return live(state.get('process', {})) and (
        not state.get('terminal_process') or live(state['terminal_process'])
    )


def display_reachable(env: dict) -> bool:
    """Probe only a captured local display, including remote-desktop X11 sessions.

    xrdp/VNC shells and user services can lack XDG_SESSION_ID or use a logind
    session of type unspecified. That is not evidence that the display is gone.
    """
    paths = []
    if match := re.fullmatch(r'(?:unix:|:)(\d+)(?:\.\d+)?', env.get('DISPLAY', '')):
        paths.append(Path('/tmp/.X11-unix') / f'X{match[1]}')
    if env.get('WAYLAND_DISPLAY') and env.get('XDG_RUNTIME_DIR'):
        runtime = Path(env['XDG_RUNTIME_DIR'])
        wayland = runtime / env['WAYLAND_DISPLAY']
        if wayland.resolve().is_relative_to(runtime.resolve()):
            paths.append(wayland)
    for path in paths:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(1)
                connection.connect(str(path))
                return True
        except OSError:
            continue
    return False


def register(root: Path, path: Path | None = None) -> bool:
    """Return false when another registered menu is alive; no effect off Linux GUI."""
    if sys.platform != 'linux' or not (os.getenv('DISPLAY') or os.getenv('WAYLAND_DISPLAY')):
        return True
    path = path or state_path()
    with lock(path):
        state = load(path)
        current = process_identity()
        if hub_alive(state) and state['process'] != current:
            return False
        state.update({'version': 1, 'uid': os.getuid(), 'root': str(root.resolve()),
                      'python': sys.executable, 'process': current, 'desired': True,
                      'terminal_process': terminal_identity(),
                      'environment': desktop_environment(), 'failures': 0, 'pending_until': 0,
                      'next_attempt': 0, 'last_error': None, 'last_launch_error': None, 'failure_paused': False})
        save(path, state)
    return True


def stop(*, reload: bool = False, path: Path | None = None) -> None:
    if sys.platform != 'linux':
        return
    path = path or state_path()
    if not path.exists():
        return
    with lock(path):
        state = load(path)
        if state.get('process') == process_identity():
            state.update({'desired': reload, 'process': {}, 'pending_until': time.time() + 60 if reload else 0})
            save(path, state)


def session_active(env: dict) -> bool:
    session = env.get('XDG_SESSION_ID')
    if not session:
        return display_reachable(env)
    try:
        result = subprocess.run(['loginctl', 'show-session', session, '-p', 'User', '-p', 'State', '-p', 'Type'],
                                capture_output=True, text=True, timeout=3)
        fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if result.returncode == 0:
            if fields.get('User') != str(os.getuid()) or fields.get('State') not in {'active', 'online'}:
                return False
            if fields.get('Type') in {'x11', 'wayland'}:
                return True
    except (OSError, subprocess.SubprocessError):
        pass
    return display_reachable(env)


def dispatch(state: dict, path: Path) -> None:
    env = {**os.environ, **state['environment']}
    env.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
    env.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path={env["XDG_RUNTIME_DIR"]}/bus')
    name = hashlib.sha256(state['root'].encode()).hexdigest()[:12]
    state['last_unit'] = f'learndl-cli-{name}-{uuid.uuid4().hex[:8]}'
    argv = ['systemd-run', '--user', '--quiet', '--collect',
            f'--unit={state["last_unit"]}',
            f'--working-directory={state["root"]}']
    for key in ENV_KEYS:
        if key in env:
            argv.append(f'--setenv={key}={env[key]}')
    argv += ['--', state['python'], str(Path(__file__).resolve()), 'launch', '--state', str(path)]
    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=5)
    if result.returncode:
        raise RuntimeError(result.stderr[-2048:] or f'systemd-run exit {result.returncode}')


def tick(path: Path | None = None, *, now: float | None = None) -> dict:
    path = path or state_path()
    if sys.platform != 'linux' or not path.exists():
        return {'status': 'unregistered'}
    now = time.time() if now is None else now
    with lock(path):
        state = load(path)
        def report(status: str) -> dict:
            state.update({'last_status': status, 'last_checked': now})
            save(path, state)
            return {'status': status}
        if not state.get('desired') or state.get('paused') or state.get('failure_paused'):
            return report('paused')
        if hub_alive(state):
            return report('alive')
        if now < state.get('pending_until', 0) or now < state.get('next_attempt', 0):
            return report('waiting-for-registration')
        if not session_active(state.get('environment', {})):
            state['last_error'] = 'Captured desktop session/display is unavailable; inspect environment and recovery.log'
            return report('waiting-for-desktop-login')
        # A dispatch is successful only after a fresh Hub registers. Account for
        # a failed acknowledgement before considering another launch.
        if state.get('pending_until'):
            state['failures'] = state.get('failures', 0) + 1
            state['last_error'] = state.get('last_launch_error') or 'CLI did not register within 60 seconds; inspect recovery.log'
            state['pending_until'] = 0
            state['next_attempt'] = now + 60 * 2 ** state['failures']
        elif state.get('failures', 0) < 3:
            try:
                dispatch(state, path)
                state['pending_until'] = now + 60
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                state['failures'] = state.get('failures', 0) + 1
                state['last_error'] = str(exc)
                state['next_attempt'] = now + 60 * 2 ** state['failures']
        if state.get('failures', 0) >= 3:
            state['failure_paused'] = True
            state['alert_pending'] = {'time': now, 'error': state['last_error']}
        report('failed' if state.get('failure_paused') else 'recovering')
        return {'status': 'failed' if state.get('failure_paused') else 'recovering',
                'failures': state.get('failures', 0), 'error': state.get('last_error')}


def deliver(email_sender, path: Path | None = None) -> bool:
    path = path or state_path()
    if not path.exists():
        return True
    with lock(path):
        state = load(path)
        notice = state.get('alert_pending')
    if not notice:
        return True
    try:
        sent = email_sender('Learndl CLI recovery paused',
                            f'Three CLI recovery attempts failed.\nProject: {state.get("root")}\n'
                            f'Error: {notice["error"]}\nState: {path}\nUse recovery resume after correcting the desktop session.',
                            attachments=[], confirmation_message='CLI recovery alert')
    except Exception:
        return False
    if sent:
        with lock(path):
            state = load(path)
            if state.get('alert_pending') == notice:
                state.pop('alert_pending', None)
                save(path, state)
    return bool(sent)


def launch(path: Path) -> None:
    state = load(path)
    if not state.get('desired') or state.get('paused') or hub_alive(state):
        return
    command = [state['python'], '-c', 'from src.api.calls.launcher import DirectCallHub; DirectCallHub.go(_recovery=True)']
    env = dict(os.environ)
    socket = env.get('WEZTERM_UNIX_SOCKET')
    if socket and not Path(socket).exists():
        env.pop('WEZTERM_UNIX_SOCKET', None)
    try:
        probe = subprocess.run(['wezterm', 'cli', 'list', '--format', 'json'], env=env,
                               capture_output=True, text=True, timeout=5)
        panes = json.loads(probe.stdout) if probe.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, ValueError):
        panes = []
    if panes:
        argv = ['wezterm', 'cli', 'spawn', '--window-id', str(panes[0]['window_id']), '--cwd', state['root'], '--', *command]
    else:
        argv = ['wezterm', 'start', '--always-new-process', '--cwd', state['root'], '--', *command]
    # Keep the cold-start GUI inside its persistent user service, not watchdog's cgroup.
    with path.with_name('recovery.log').open('a', buffering=1) as output:
        output.write(f'{time.time()}: launching {argv!r}\n')
        try:
            subprocess.run(argv, env=env, check=True, stdout=output, stderr=output)
        except (OSError, subprocess.SubprocessError) as exc:
            output.write(f'{time.time()}: {exc}\n')
            with lock(path):
                latest = load(path)
                latest['last_launch_error'] = str(exc)
                save(path, latest)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'pause', 'resume', 'login', 'launch'])
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    path = args.state or state_path()
    if args.action == 'launch':
        launch(path)
        return
    if args.action == 'status':
        print(json.dumps(load(path), indent=2))
        return
    with lock(path):
        state = load(path)
        if args.action == 'login':
            if not state:
                return  # Login never opts an unregistered project into recovery.
            state['environment'] = desktop_environment()
        else:
            state['paused'] = args.action == 'pause'
            if args.action == 'resume':
                state.update({'failure_paused': False, 'failures': 0, 'pending_until': 0, 'next_attempt': 0})
        save(path, state)
    if args.action in {'login', 'resume'}:
        print(tick(path))


if __name__ == '__main__':
    main()
