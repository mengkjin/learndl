"""Windows: WezTerm — ``cli spawn`` when a GUI socket exists, else ``wezterm start``."""

from __future__ import annotations

import base64
import glob
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ...preference import WINDOWS_WEZTERM_NEW
from ...util.basic import BasicOpener
from ...util import process
from .verify import WezTermVerifier

# Optional: focus an existing WezTerm GUI window (e.g. after ``start``).
_PS_FOCUS = r"""
$proc = $null
foreach ($n in @('wezterm-gui', 'wezterm', 'WezTerm')) {
  $proc = Get-Process -Name $n -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
  if ($proc) { break }
}
if (-not $proc) { exit 1 }
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class W32 {
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
}
'@
[void][W32]::ShowWindow($proc.MainWindowHandle, 9)
[void][W32]::SetForegroundWindow($proc.MainWindowHandle)
exit 0
"""


def _windows_process_exe_path(pid: int) -> str | None:
    """Executable path for ``pid``, or ``None`` if inaccessible / not running."""
    if sys.platform != "win32" or pid <= 0:
        return None
    import ctypes
    from ctypes import wintypes

    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(4096)
        n = wintypes.DWORD(len(buf))
        if not k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
            return None
        return buf.value
    finally:
        k.CloseHandle(h)


def _pid_from_gui_sock_basename(name: str) -> int | None:
    if name.startswith("gui-sock-"):
        tail = name[len("gui-sock-") :]
        if tail.isdigit():
            return int(tail)
    return None


def _socket_targets_live_wezterm(path: str) -> bool:
    """
    Whether ``path`` is usable for ``cli spawn``.

    ``OpenProcess`` alone is not enough: Windows reuses PIDs, so a stale ``gui-sock-<pid>``
    can still "exist" as another process. Require the executable path to mention wezterm.
    """
    if not os.path.exists(path):
        return False
    pid = _pid_from_gui_sock_basename(os.path.basename(path))
    if pid is None:
        return True
    exe = _windows_process_exe_path(pid)
    if not exe:
        return False
    return "wezterm" in exe.lower()


def _debug_wezterm_socket(msg: str) -> None:
    if os.environ.get("SHELL_OPENER_DEBUG_WEZTERM"):
        print(msg, flush=True)


def discover_wezterm_gui_socket() -> str | None:
    """
    Absolute path to a live WezTerm GUI IPC socket (``gui-sock-<pid>`` in ``RUNTIME_DIR``).

    WezTerm stores these under the same layout as Unix: ``~/.local/share/wezterm/`` on all
    OSes (not ``wezterm-gui-sock-*`` in ``%TEMP%``, which was a mistaken pattern). See
    ``wezterm-client`` ``discover_gui_socks`` and issues #3374, #4456.
    """
    if sys.platform != "win32":
        return None
    env_sock = os.environ.get("WEZTERM_UNIX_SOCKET")
    if env_sock:
        env_sock = os.path.abspath(env_sock)
        if _socket_targets_live_wezterm(env_sock):
            return env_sock
        _debug_wezterm_socket(f"ignore WEZTERM_UNIX_SOCKET (stale or not wezterm): {env_sock!r}")

    roots: list[str] = []
    roots.append(str(Path.home() / ".local" / "share" / "wezterm"))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(os.path.join(local, "wezterm"))
    tmp = os.environ.get("TEMP") or os.environ.get("TMP") or ""
    if tmp:
        roots.append(tmp)

    matches: list[str] = []
    for root in roots:
        if not root:
            continue
        if os.path.isdir(root):
            try:
                for name in os.listdir(root):
                    if name.startswith("gui-sock-") or name.startswith("wezterm-gui-sock-"):
                        p = os.path.join(root, name)
                        if os.path.exists(p):
                            matches.append(os.path.abspath(p))
            except OSError:
                pass
        for pat in ("gui-sock-*", "wezterm-gui-sock-*"):
            for p in glob.glob(os.path.join(root, pat)):
                if os.path.exists(p):
                    matches.append(os.path.abspath(p))

    if not matches:
        return None
    live: list[str] = []
    for p in dict.fromkeys(matches):
        if _socket_targets_live_wezterm(p):
            live.append(p)
            spid = _pid_from_gui_sock_basename(os.path.basename(p))
            if spid is not None:
                _debug_wezterm_socket(
                    f"accept gui socket {p!r} pid={spid} exe={_windows_process_exe_path(spid)!r}"
                )
            else:
                _debug_wezterm_socket(f"accept socket {p!r}")
        else:
            _debug_wezterm_socket(f"skip gui socket (stale PID or exe not wezterm): {p!r}")
    if not live:
        return None
    return max(live, key=lambda p: os.stat(p).st_mtime)


def activate_wezterm() -> bool:
    """
    Bring a WezTerm window to the foreground (Windows).

    Uses PowerShell + ``user32`` (no extra dependencies). Returns False if no WezTerm
    window with a main handle was found or activation failed.
    """
    if sys.platform != "win32":
        return False
    try:
        enc = base64.b64encode(_PS_FOCUS.encode("utf-16-le")).decode("ascii")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.run(
            ["powershell", "-NoProfile", "-NoLogo", "-EncodedCommand", enc],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15.0,
            creationflags=flags,
        )
        return True
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
    ):
        return False


def bring_wezterm_to_foreground_soon(*, delay_s: float = 0.25) -> None:
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
        if new_on is None:
            new_on = WINDOWS_WEZTERM_NEW
        inner = command
        if title is not None:
            title = title.replace('"', "'")
            inner = f"wezterm cli set-tab-title {title} & {inner}"
        tail = ["--", "cmd.exe", "/k", inner]
        spawn_env: dict[str, str] | None = None

        match new_on:
            case "tab":
                # Inside a pane: env already has context for ``cli spawn``.
                # Outside: discover ``wezterm-gui-sock-*`` in TEMP and set ``WEZTERM_UNIX_SOCKET``.
                # Otherwise ``start --new-tab`` often opens another window (no session binding).
                in_pane = bool(os.environ.get("WEZTERM_PANE"))
                sock = None if in_pane else discover_wezterm_gui_socket()
                if in_pane or sock:
                    args = ["wezterm", "cli", "spawn"]
                    if cwd:
                        args.extend(["--cwd", cwd])
                    args.extend(tail)
                    if sock:
                        spawn_env = {**os.environ, "WEZTERM_UNIX_SOCKET": sock}
                else:
                    args = ["wezterm", "start"]
                    if cwd:
                        args.extend(["--cwd", cwd])
                    args.extend(tail)
                process.popen_detached(
                    args,
                    env=spawn_env,
                    windows_detached_process=False,
                    windows_create_no_window=False,
                )
                bring_wezterm_to_foreground_soon()
            case "window" | "workspace":
                args = ['wezterm', "start", "--always-new-process"]
                if cwd:
                    args.extend(["--cwd", cwd])
                args.extend(tail)
                process.popen_detached(args)
                bring_wezterm_to_foreground_soon()
            case _:
                raise ValueError(f"Invalid new_on: {new_on}")
