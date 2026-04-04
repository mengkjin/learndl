"""cmux Unix-socket IPC: workspace / window / surface targets."""

from __future__ import annotations

import re
import shlex
import threading
import time
import subprocess
from typing import Literal

from terminal_opener.util import process
import terminal_opener.preference as preference

from .cli import CmuxCli
from .verify import CmuxVerifier


"""rename-tab [--workspace <id|ref>] [--tab <id|ref>] [--surface <id|ref>] <title>"""
def popup_cmux() -> None:
    subprocess.run(["osascript", "-e", 'tell application "cmux" to activate'],check=False)

def start_cmux():
    """Start cmux and return the window id."""
    ping_ok , _= CmuxCli.cmux_ping()
    if ping_ok:
        return

    process.popen_detached(["open", "-a", "cmux"])
    deadline = time.monotonic() + preference.MACOS_CMUX_COLD_START_DEADLINE
    while time.monotonic() < deadline:
        ping_ok , detail = CmuxCli.cmux_ping()
        if ping_ok:
            return
        time.sleep(preference.MACOS_CMUX_PING_INTERVAL)

    error_messages = [f"ping cmux failed after {preference.MACOS_CMUX_COLD_START_DEADLINE} seconds: {detail or 'no detail'}"]
    if any(k in detail.lower()
        for k in ("socket","ancestr","denied","forbidden","not allowed", "write to socket","configure socket",)
    ):
        error_messages.append(
            "cmux IPC failed (%s). If cmux is open but commands never run, set "
            "CMUX_SOCKET_MODE=allowAll, restart cmux, then retry (default socket only allows "
            "children of cmux terminals)."
        )
    raise RuntimeError('\n'.join(error_messages))

def cmux_get_workspaces() -> set[str]:
    ret = CmuxCli.cmux_json("list-workspaces")
    return set([info['ref'] for info in ret['workspaces']])

def cmux_get_surfaces() -> set[str]:
    ret = CmuxCli.cmux_json("list-panels")
    return set([info['ref'] for info in ret['surfaces']])

def cmux_new_window(title: str | None = None , cwd: str | None = None , focus: bool = True) -> str:
    win = CmuxCli.cmux_json("new-window").removeprefix('OK ')
    if focus:
        CmuxCli.cmux("focus-window", "--window" , win)
    ref = CmuxCli.cmux_json('current-workspace')['workspace_id']
    if title:
        CmuxCli.cmux('rename-workspace', '--workspace', ref, title)
    if cwd:
        CmuxCli.cmux('send', '--workspace', ref, f"cd {cwd}\n")
    return ref

def cmux_new_workspace(title: str | None = None, cwd: str | None = None , focus: bool = True):
    new_args = []
    if title:
        new_args.extend(['--name', title])
    if cwd:
        new_args.extend(['--cwd', cwd])
    ref = CmuxCli.cmux_json('new-workspace', *new_args).removeprefix('OK ')
    if title:
        CmuxCli.cmux('rename-tab', '--workspace', ref, title)
    if focus:
        CmuxCli.cmux('select-workspace', '--workspace', ref)
    return ref

def cmux_new_surface(title: str | None = None, cwd: str | None = None , focus: bool = True) -> str:
    ref = CmuxCli.cmux_json("new-surface")['surface_ref']
    if focus:
        CmuxCli.cmux("focus-panel", "--panel" , ref)
    if title:
        CmuxCli.cmux("rename-tab", "--surface" , ref, title)
    if cwd:
        CmuxCli.cmux('send', '--surface', ref, f"cd {cwd}\n")
    return ref

def run_in_new_window(cwd: str, command: str , * , title : str | None = None):
    ref = cmux_new_window(title = title)
    line = f"cd {shlex.quote(cwd)} && {command}\n"
    CmuxCli.cmux("send", "--workspace", ref , line)
    
def run_in_new_workspace(cwd: str, command: str , * , title : str | None = None):
    ref = cmux_new_workspace(title = title)
    line = f"cd {shlex.quote(cwd)} && {command}\n"
    CmuxCli.cmux('send', '--workspace', ref, line)

def run_in_new_surface(cwd: str, command: str , * , title : str | None = None):
    surface = cmux_new_surface(title=title)
    line = f"cd {shlex.quote(cwd)} && {command}\n"
    CmuxCli.cmux("send", "--surface", surface, line)

def cmux_run(
    cwd: str, command: str, * , 
    target: Literal["workspace", "window", "surface"] | None = None, 
    title: str | None = None
) -> None:
    if target is None:
        if preference.MACOS_CMUX_NEW in ["workspace", "window", "surface"]:
            target = preference.MACOS_CMUX_NEW # type: ignore
        else:
            raise ValueError(f"Invalid target: {preference.MACOS_CMUX_NEW}")
    kwargs = {'cwd': cwd,'command': command,'title': title,}
    start_cmux()
    if not CmuxCli.cmux_json("list-windows"):
        target = "window" 
    
    if target == "window":
        run_in_new_window(**kwargs)
    elif target == "workspace":
        run_in_new_workspace(**kwargs)
    elif target == "surface":
        run_in_new_surface(**kwargs)
    else:
        raise ValueError(f"Invalid target: {target}")
    popup_cmux()

def guess_command_title(command: str) -> str | None:
    """
    extract .py filename from command:
        python3 any/path/name.py
        python.exe any/path/name.py
        uv run any/path/name.py
        python C:\\my folder\\app.py   #support space in path
    return filename (e.g. name.py), return None if not matched
    """
    pattern = re.compile(
        r'(?:python[\d.]*(?:\.exe)?|uv\s+run)\s+(.*?\.py)(?=[\s;]|$)',
        re.IGNORECASE
    )
    match = pattern.search(command)
    if not match:
        return None
    
    full_path = match.group(1)
    normalized = full_path.replace('\\', '/')
    filename = normalized.split('/')[-1]
    return filename

class CmuxOpener:

    @classmethod
    def run(cls ,
        cwd: str,
        command: str,
        *,
        target: Literal["workspace", "window", "surface"] | None = None,
    ) -> None:
        """Run cmux IPC in a non-daemon thread (survives short-lived parent processes)."""
        if not CmuxVerifier.available():
            raise RuntimeError("cmux is not available")
        
        title = guess_command_title(command)
        t = threading.Thread(
            target=cmux_run,
            args=(cwd, command),
            kwargs={'title': title, 'target': target},
            name="cmux-opener",
            daemon=False,
        )
        t.start()
