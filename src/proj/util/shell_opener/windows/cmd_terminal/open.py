"""Windows: ``cmd`` in a new console, or Windows Terminal tab/window when ``wt.exe`` is available."""

from __future__ import annotations

import ctypes
import os
import shutil
from ctypes import wintypes

from ...preference import WINDOWS_CMD_NEW
from ...util.basic import BasicOpener
from ...util.process import popen_detached, popen_detached_shell_windows
from .verify import CmdTerminalVerifier

TH32CS_SNAPPROCESS = 0x00000002
SW_RESTORE = 9


def _cmd_quoted(s: str) -> str:
    """Double-quote for ``cmd.exe`` metasyntax (internal ``"`` → ``""``)."""
    return '"' + s.replace('"', '""') + '"'


def raise_recent_windows_terminal() -> bool:
    """
    Best-effort: find a visible top-level window owned by ``WindowsTerminal.exe``,
    restore it if minimized, and bring it to the foreground.

    Returns False if Windows Terminal is not running or the operation fails.
    """
    if os.name != "nt":
        return False

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            # Win32 ULONG_PTR (pointer-sized); c_size_t matches on Windows.
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    pids: set[int] = set()
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap in (-1, 0xFFFFFFFF):
        return False
    try:
        pe = PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if not kernel32.Process32FirstW(snap, ctypes.byref(pe)):
            return False
        while True:
            name = pe.szExeFile.lower()
            if name == "windowsterminal.exe":
                pids.add(int(pe.th32ProcessID))
            if not kernel32.Process32NextW(snap, ctypes.byref(pe)):
                break
    finally:
        kernel32.CloseHandle(snap)

    if not pids:
        return False

    target: list[int | None] = [None]

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_proc(hwnd: wintypes.HWND, _lparam: wintypes.LPARAM) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) in pids:
            target[0] = int(hwnd)
            return False
        return True

    user32.EnumWindows(enum_proc, 0)
    hwnd = target[0]
    if hwnd is None:
        return False

    user32.ShowWindow(hwnd, SW_RESTORE)
    return bool(user32.SetForegroundWindow(hwnd))


class CmdTerminalOpener(BasicOpener):
    def available(self) -> bool:
        return CmdTerminalVerifier.available()

    def run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        title: str | None = None,
        new_on: str | None = None,
    ) -> None:
        assert self._available, f"{self.__class__.__name__} is not available"

        if new_on is None:
            new_on = WINDOWS_CMD_NEW

        match new_on:
            case "window" | "workspace" | "tab":
                pass
            case _:
                raise ValueError(f"Invalid new_on: {new_on}")

        def _inner_for_start() -> str:
            parts: list[str] = []
            if cwd:
                parts.append(f"cd /d {_cmd_quoted(cwd)}")
            parts.append(command)
            return " & ".join(parts)

        wt = shutil.which("wt")
        if wt:
            if new_on == "tab":
                if raise_recent_windows_terminal():
                    win_target = "last"
                else:
                    win_target = "-1"
            else:
                win_target = "-1"

            args: list[str] = [wt, "-w", win_target, "nt"]
            if title is not None:
                args.extend(["--title", title])
            if cwd:
                args.extend(["-d", cwd])
            comspec = os.environ.get("ComSpec") or "cmd.exe"
            # ``wt -d`` sets the tab working directory; avoid duplicating ``cd``.
            args.extend([comspec, "/k", command])
            popen_detached(args)
            return

        # Classic ``start`` + ``conhost``: no tabs; always a new console window.
        inner_cmd = _inner_for_start()
        title_for_start = '""' if title is None else _cmd_quoted(title)
        escaped = inner_cmd.replace('"', '""')
        shell_cmd = f"start {title_for_start} cmd /k \"{escaped}\""
        popen_detached_shell_windows(shell_cmd)
