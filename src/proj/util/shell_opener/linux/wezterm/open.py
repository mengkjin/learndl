"""Linux: WezTerm via ``wezterm cli spawn`` when a GUI is up, else ``wezterm start`` (cold)."""

from __future__ import annotations

import glob
import os
import sys
import shlex
import stat
import subprocess
import threading
import time

from ...preference import LINUX_WEZTERM_NEW
from ...util import process
from ...util.basic import BasicOpener
from .verify import WezTermVerifier


def _debug_wezterm_socket(msg: str) -> None:
    if os.environ.get("SHELL_OPENER_DEBUG_WEZTERM"):
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()


def _wezterm_runtime_dir() -> str:
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not base:
        base = f"/run/user/{os.getuid()}"
    return os.path.join(base, "wezterm")


def _pid_from_gui_sock_basename(name: str) -> int | None:
    if name.startswith("gui-sock-"):
        tail = name[len("gui-sock-") :]
        if tail.isdigit():
            return int(tail)
    return None


def _linux_socket_targets_live_wezterm(path: str) -> bool:
    """Whether ``path`` is a live WezTerm GUI socket (PID alive and exe is wezterm)."""
    try:
        st = os.stat(path)
    except OSError:
        return False
    if not stat.S_ISSOCK(st.st_mode):
        return False
    pid = _pid_from_gui_sock_basename(os.path.basename(path))
    if pid is None:
        return True
    if not os.path.isdir(f"/proc/{pid}"):
        return False
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return False
    return "wezterm" in os.path.basename(exe).lower()


def discover_wezterm_gui_socket() -> str | None:
    """
    Path to a live WezTerm GUI IPC socket (``gui-sock-<pid>`` under the runtime dir).

    ``wezterm cli`` requires this; without a running GUI, use ``wezterm start`` instead.
    """
    candidates: list[str] = []

    env_sock = os.environ.get("WEZTERM_UNIX_SOCKET")
    if env_sock:
        env_sock = os.path.abspath(env_sock)
        if _linux_socket_targets_live_wezterm(env_sock):
            return env_sock
        _debug_wezterm_socket(f"ignore WEZTERM_UNIX_SOCKET (stale or not wezterm): {env_sock!r}")

    rt = _wezterm_runtime_dir()
    if os.path.isdir(rt):
        try:
            for name in os.listdir(rt):
                if name.startswith("gui-sock-"):
                    p = os.path.join(rt, name)
                    if os.path.exists(p):
                        candidates.append(os.path.abspath(p))
        except OSError:
            pass
    for p in glob.glob(os.path.join(rt, "gui-sock-*")):
        if os.path.exists(p):
            candidates.append(os.path.abspath(p))

    if not candidates:
        return None

    live: list[str] = []
    for p in dict.fromkeys(candidates):
        if _linux_socket_targets_live_wezterm(p):
            live.append(p)
            spid = _pid_from_gui_sock_basename(os.path.basename(p))
            if spid is not None:
                _debug_wezterm_socket(f"accept gui socket {p!r} pid={spid}")
        else:
            _debug_wezterm_socket(f"skip gui socket (stale or not wezterm): {p!r}")

    if not live:
        return None
    return max(live, key=lambda x: os.stat(x).st_mtime)


def activate_wezterm() -> bool:
    """
    Bring WezTerm to the foreground (Linux, X11 via ``wmctrl``).

    Returns True if a WezTerm window was found and activated; False if ``wmctrl`` is
    missing, no matching window exists, or activation failed.
    """
    try:
        output = subprocess.check_output(["wmctrl", "-lx"], stderr=subprocess.STDOUT, timeout=5.0).decode()
        lines = [line for line in output.splitlines() if "wezterm" in line.lower()]
        if not lines:
            return False
        window_id = lines[-1].split()[0]
        subprocess.run(
            ["wmctrl", "-i", "-a", window_id],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, subprocess.TimeoutExpired, OSError):
        return False


def bring_wezterm_to_foreground_soon(*, delay_s: float = 0.35) -> None:
    """After spawning a tab, briefly wait then focus a WezTerm window (non-blocking)."""

    def _run() -> None:
        time.sleep(delay_s)
        activate_wezterm()

    threading.Thread(target=_run, daemon=True).start()


class WezTermOpener(BasicOpener):
    def available(self) -> bool:
        return WezTermVerifier.available()

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        title: str | None = None,
        new_on: str | None = None,
        **kwargs,
    ) -> None:
        assert self._available, f"{self.__class__.__name__} is not available"
        command = f"{command}; exec bash"
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        if title is not None:
            command = f'echo -ne "\\033]0;{title}\\a"; {command}'
        if new_on is None:
            new_on = LINUX_WEZTERM_NEW

        sock = discover_wezterm_gui_socket()
        spawn_env: dict[str, str] | None = None
        if sock is not None:
            spawn_env = {**os.environ, "WEZTERM_UNIX_SOCKET": sock}

        # Cold start: no live GUI — ``wezterm cli`` cannot connect; open first window via ``start``.
        if sock is None:
            args: list[str] = ["wezterm", "start"]
            if cwd:
                args.extend(["--cwd", cwd])
            args.extend(["--", "bash", "-lc", command])
            process.popen_detached(args)
            bring_wezterm_to_foreground_soon()
            return

        match new_on:
            case "window" | "workspace":
                args = ["wezterm", "cli", "spawn", "--new-window"]
            case "tab":
                activate_wezterm()
                args = ["wezterm", "cli", "spawn"]
            case _:
                raise ValueError(f"Invalid new_on: {new_on}")
        if cwd:
            args.extend(["--cwd", cwd])
        args.extend(["--", "bash", "-lc", command])
        process.popen_detached(args, env=spawn_env)
        if new_on == "tab":
            bring_wezterm_to_foreground_soon()
