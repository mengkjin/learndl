"""cmux Unix-socket IPC: workspace / window / surface targets."""

from __future__ import annotations

import shlex
import threading
import time
import subprocess

from typing import Literal
from dataclasses import dataclass

from .cli import CmuxCli
from .verify import CmuxVerifier
from ...util import process , BasicOpener
from ... import preference

_workspace_refs: dict[str, str] = {
    
}

@dataclass
class CmuxRefs:
    window_id: str
    workspace_ref: str
    surface_ref: str

    def focus(self , window : bool = False , surface : bool = True) -> None:
        if window:
            CmuxCli.cmux('focus-window', '--window', self.window_id)
        if surface:
            CmuxCli.cmux('select-workspace', '--workspace', self.workspace_ref)
            CmuxCli.cmux('focus-panel', '--panel', self.surface_ref , '--workspace', self.workspace_ref)

    def cwd(self, cwd: str | None = None) -> None:
        if cwd:
            CmuxCli.cmux('send', '--workspace', self.workspace_ref , '--surface', self.surface_ref, f'cd {shlex.quote(cwd)}\n')

    def send(self, command: str , * , cwd: str | None = None) -> None:
        self.cwd(cwd)
        CmuxCli.cmux("send", "--workspace", self.workspace_ref , "--surface", self.surface_ref, f'{command}\n')

    def rename(self , title: str | None = None , where: Literal["window", "workspace", "surface", "all"] = "surface") -> None:
        if not title:
            return
        match where:
            case "window":
                CmuxCli.cmux('rename-window', '--window', self.window_id, title)
            case "workspace":
                CmuxCli.cmux('rename-workspace', '--workspace', self.workspace_ref, title)
            case "surface":
                CmuxCli.cmux('rename-tab', '--surface', self.surface_ref, '--workspace', self.workspace_ref, title)
            case "all":
                self.rename(title=title, where="window")
                self.rename(title=title, where="workspace")
                self.rename(title=title, where="surface")

def popup_cmux() -> None:
    """
    Activate the cmux application to the foreground.
    """
    subprocess.run(["osascript", "-e", 'tell application "cmux" to activate'],check=False)

def start_cmux():
    """Start cmux if ping failed. But even if ping passed, likely no window is opened."""
    ping_ok , detail = CmuxCli.cmux_ping()
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

def cmux_current_window_id() -> str:
    return CmuxCli.cmux("current-window").stdout.strip()

def cmux_new_window_id() -> str:
    return CmuxCli.cmux("new-window").stdout.strip().removeprefix('OK ')

def cmux_current_refs(window_id: str | None = None , workspace_ref: str | None = None) -> CmuxRefs:
    """Get the current surface reference."""
    assert window_id is None or workspace_ref is None, "window_id and workspace_ref must not be set at the same time"
    if window_id:
        CmuxCli.cmux_json("focus-window" , "--window" , window_id)
    else:
        window_id = cmux_current_window_id()

    if workspace_ref:
        ret = CmuxCli.cmux_json("list-panels" , "--workspace" , workspace_ref)
    else:
        ret = CmuxCli.cmux_json("list-panels")
        workspace_ref = ret['workspace_ref']
    if [x for x in ret['surfaces'] if x['focused']]:
        surface_ref = [x['ref'] for x in ret['surfaces'] if x['focused']][-1]
    else:
        surface_ref = ret['surfaces'][-1]['ref']
    assert workspace_ref 
    return CmuxRefs(window_id=window_id, workspace_ref=workspace_ref, surface_ref=surface_ref)

def cmux_new_window(title: str | None = None , focus: bool = True , as_workspace: str | None = None) -> CmuxRefs:
    """Create a new window and return the workspace reference."""
    win_id = CmuxCli.cmux_json("new-window").removeprefix('OK ')

    time.sleep(0.1)
    refs = cmux_current_refs(window_id=win_id)
    refs.rename(title=title or as_workspace, where="all")
    
    if as_workspace:
        assert as_workspace not in _workspace_refs, f"Workspace {as_workspace} already exists : {_workspace_refs[as_workspace]}"
        _workspace_refs[as_workspace] = refs.workspace_ref
    return refs

def cmux_new_workspace(title: str | None = None, focus: bool = True , as_workspace: str | None = None) -> CmuxRefs:
    """Create a new workspace and return the workspace reference."""
    new_args = []
    if title:
        new_args.extend(['--name', title])
    workspace_ref = CmuxCli.cmux_json('new-workspace', *new_args).removeprefix('OK ')
    
    time.sleep(0.1)
    refs = cmux_current_refs(workspace_ref=workspace_ref)
    refs.rename(title=title or as_workspace, where="surface")
    refs.focus(surface=focus, window=False)
    if as_workspace:
        assert as_workspace not in _workspace_refs, f"Workspace {as_workspace} already exists : {_workspace_refs[as_workspace]}"
        _workspace_refs[as_workspace] = refs.workspace_ref
    return refs

def cmux_new_surface(title: str | None = None, focus: bool = True , from_workspace: str | None = None) -> CmuxRefs:
    """Create a new surface and return the surface reference."""
    refs = None
    if from_workspace and from_workspace in _workspace_refs:
        try:
            workspace_ref = _workspace_refs[from_workspace]
            ret = CmuxCli.cmux_json("new-surface" , "--workspace" , workspace_ref)

            time.sleep(0.1)
            refs = CmuxRefs(window_id=cmux_current_window_id(), workspace_ref=ret['workspace_ref'], surface_ref=ret['surface_ref'])
        except Exception:
            _workspace_refs.pop(from_workspace)
    if not refs:
        if not from_workspace:
            ret = CmuxCli.cmux_json("new-surface")

            time.sleep(0.1)
            refs = CmuxRefs(window_id=cmux_current_window_id(), workspace_ref=ret['workspace_ref'], surface_ref=ret['surface_ref'])
        else:
            refs = cmux_new_workspace(title=from_workspace , as_workspace=from_workspace)
    
    refs.rename(title=title, where="surface")
    refs.focus(surface=focus, window=True)
    return refs

def run_in_new_window(command: str , * , cwd: str | None = None, title : str | None = None, 
                      as_workspace: str | None = None):
    cmux_new_window(title = title , as_workspace=as_workspace).send(command , cwd=cwd)

def run_in_new_workspace(command: str , * , cwd: str | None = None, title : str | None = None , 
                         as_workspace: str | None = None):
    cmux_new_workspace(title = title, as_workspace=as_workspace).send(command , cwd=cwd)

def run_in_new_surface(command: str , * , cwd: str | None = None, title : str | None = None , 
                       from_workspace: str | None = None):
    cmux_new_surface(title=title, from_workspace=from_workspace).send(command , cwd=cwd)

def cmux_run(
    command: str, * , 
    cwd: str | None = None, 
    new_on: str | None = None, 
    title: str | None = None,
    as_workspace: str | None = None,
    from_workspace: str | None = None,
) -> None:
    
    kwargs = {'cwd': cwd,'command': command,'title': title,}
    start_cmux()
    if new_on is None:
        new_on = preference.MACOS_CMUX_NEW
    if not CmuxCli.cmux_json("list-windows"):
        new_on = "window" 
    match new_on:
        case "window":
            run_in_new_window(**kwargs , as_workspace=as_workspace)
        case "workspace":
            run_in_new_workspace(**kwargs, as_workspace=as_workspace)
        case "tab":
            run_in_new_surface(**kwargs, from_workspace=from_workspace)
        case _:
            raise ValueError(f"Invalid new_on: {new_on}")
    popup_cmux()

class CmuxOpener(BasicOpener):
    def available(self) -> bool:
        return CmuxVerifier.available()

    def run(self ,
        command: str,
        * ,
        cwd: str | None = None,
        new_on: str | None = None,
        title: str | None = None,
        as_workspace: str | None = None,
        from_workspace: str | None = None,
        **kwargs
    ) -> None:
        """Run cmux IPC in a non-daemon thread (survives short-lived parent processes)."""
        assert self._available , f"{self.__class__.__name__} is not available"
        t = threading.Thread(
            target=cmux_run,
            args=(command,),
            kwargs={'cwd': cwd, 'title': title, 'new_on': new_on , 'as_workspace': as_workspace, 'from_workspace': from_workspace},
            name="cmux-opener",
            daemon=False,
        )
        t.start()
