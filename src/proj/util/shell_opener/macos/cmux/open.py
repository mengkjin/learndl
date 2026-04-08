"""cmux Unix-socket IPC: workspace / window / pane targets."""

from __future__ import annotations

import shlex
import threading
import time
import subprocess

from typing import Literal

from .cli import CmuxCli
from .verify import CmuxVerifier
from ...util import process , BasicOpener
from ... import preference

class CmuxTree:
    SavedWorkspace : dict[str , str] = {}
    def __init__(self, windows: list[CmuxWindow]):
        self.windows = windows

    def __repr__(self):
        return f"Tree(windows={self.windows})"

    def flattern(self):
        surfaces : list[tuple[CmuxWindow , CmuxWorkspace , CmuxSurface]] = []
        for window in self.windows:
            for workspace in window.workspaces:
                for surface in workspace.surfaces:
                    surfaces.append((window , workspace , surface))
        return surfaces

    @property
    def current_window(self):
        return [window for window in self.windows if window.active][-1]

    @property
    def current_workspace(self):
        return [workspace for workspace in self.current_window.workspaces if workspace.selected][-1]

    @property
    def current_surface(self):
        return [surface for surface in self.current_workspace.surfaces if surface.focused][-1]

    @property
    def workspaces(self):
        return [workspace for window in self.windows for workspace in window.workspaces]

    @property
    def surfaces(self):
        return [surface for workspace in self.workspaces for surface in workspace.surfaces]

    @classmethod
    def from_cmux(cls):
        tree = CmuxCli.cmux_json("--id-format" , "both" , "tree" , "--all")
        windows : list[CmuxWindow] = []
        for window in tree['windows']:
            w = CmuxWindow(**window)
            for workspace in window['workspaces']:
                ws = w.add_child(workspace)
                surfaces = [surface for pane in workspace['panes'] for surface in pane['surfaces']]
                for surface in surfaces:
                    ws.add_child(surface)
            windows.append(w)
        self = cls(windows)
        self.refresh_saved_workspace()
        return self

    def refresh_saved_workspace(self):
        workspace_refs = [workspace.ref for workspace in self.workspaces]
        for title , ref in self.SavedWorkspace.items():
            if ref not in workspace_refs:
                self.SavedWorkspace.pop(title)
        return self

    def refresh(self):
        new_tree = self.from_cmux()
        self.windows = new_tree.windows
        self.refresh_saved_workspace()
        return self   

    def get_workspace(self , workspace_ref: str | None = None , from_workspace: str | None = None) -> CmuxWorkspace | None:
        assert workspace_ref is None or from_workspace is None, "workspace_ref and from_workspace must not be set at the same time"
        assert workspace_ref or from_workspace, "workspace_ref or from_workspace must be set"
        if workspace_ref:
            workspaces = [workspace for window in self.windows for workspace in window.workspaces if workspace.ref == workspace_ref]
        else:
            if from_workspace in self.SavedWorkspace:
                workspace_ref = self.SavedWorkspace[from_workspace]
                workspaces = [workspace for window in self.windows for workspace in window.workspaces if workspace.ref == workspace_ref] 
            else:
                workspaces = []
            if not workspaces:
                workspaces = [workspace for workspace in self.current_window.workspaces if workspace.kwargs.get('title') == from_workspace]
        return workspaces[-1] if workspaces else None

    def get_surface(self , window_id: str | None = None , workspace_ref: str | None = None) -> CmuxSurface | None:
        """Get the current surface reference."""
        assert window_id is None or workspace_ref is None, "window_id and workspace_ref must not be set at the same time"
        if window_id:
            workspaces = [workspace for window in self.windows if window.id == window_id for workspace in window.workspaces]
        else:
            workspaces = self.current_window.workspaces
        
        if workspace_ref:
            surfaces = [surface for workspace in workspaces if workspace.ref == workspace_ref for surface in workspace.surfaces]
        else:
            surfaces = self.current_workspace.surfaces

        return surfaces[-1] if surfaces else None

    @classmethod
    def new_window(cls , title : str | None = None , focus: bool = True , as_workspace: str | None = None) -> CmuxSurface:
        """Create a new window and return the activesurface"""
        window_id = CmuxCli.cmux_json("new-window").removeprefix('OK ')
        tree = cls.from_cmux()
        window = [window for window in tree.windows if window.id == window_id][-1]
        if focus:
            window.focus()
        workspace = window.workspaces[-1]
        if as_workspace:
            workspace.save_workspace(as_workspace)
        return workspace.surfaces[-1].rename(title , where = 'all')

    @classmethod
    def new_workspace(cls , title: str | None = None, focus: bool = True , as_workspace: str | None = None) -> CmuxSurface:
        """Create a new workspace and return the workspace reference."""
        new_args = []
        if title:
            new_args.extend(['--name', title])
        workspace_ref = CmuxCli.cmux_json('new-workspace', *new_args).removeprefix('OK ')
        tree = cls.from_cmux()
        surface = tree.get_surface(workspace_ref=workspace_ref)
        assert surface is not None, f"No surface found for workspace_ref={workspace_ref}"
        return surface.rename(title=title or as_workspace, where="surface").focus(surface=focus, window=False).save_workspace(title=as_workspace)

    @classmethod
    def new_surface(cls , title: str | None = None, focus: bool = True , from_workspace: str | None = None) -> CmuxSurface:
        """Create a new surface and return the surface reference."""
        tree = cls.from_cmux()
        workspace = tree.get_workspace(from_workspace=from_workspace)
        if workspace:
            surface = workspace.new_surface(title=title)
        else:
            surface = tree.new_workspace(title=from_workspace , as_workspace=from_workspace).rename(title=title, where="surface")
        return surface.rename(title=title, where="surface").focus(surface=focus, window=True)

class CmuxWindow:
    def __init__(self , id: str , active: bool , **kwargs):
        self.id = id
        self.active = active
        self.kwargs = kwargs
        self.workspaces : list[CmuxWorkspace] = []

    def __repr__(self):
        return f"Window(id={self.id}, active={self.active})"

    def add_child(self, input : dict):
        workspace = CmuxWorkspace(window = self , **input)
        self.workspaces.append(workspace)
        return workspace

    def focus(self):
        CmuxCli.cmux('focus-window', '--window', self.id)

    
class CmuxWorkspace:
    def __init__(self , ref: str , selected: bool , title: str , window: CmuxWindow , **kwargs):
        self.ref = ref
        self.selected = selected
        self.title = title
        self.window = window
        self.kwargs = kwargs
        self.surfaces : list[CmuxSurface] = []

    def __repr__(self):
        return f"Workspace(ref={self.ref}, selected={self.selected}, title={self.title})"

    def add_child(self, input : dict):
        surface = CmuxSurface(workspace = self , window = self.window , **input)
        self.surfaces.append(surface)
        return surface

    def save_workspace(self , title : str | None = None):
        if title:
            CmuxTree.SavedWorkspace[title] = self.ref

    def select(self):
        CmuxCli.cmux('select-workspace', '--workspace', self.ref)

    def new_surface(self , title: str | None = None) -> CmuxSurface:
        ret = CmuxCli.cmux_json("new-surface" , "--workspace" , self.ref)
        if title:
            CmuxCli.cmux('rename-tab', '--surface', ret['surface_ref'], '--workspace', self.ref, title)
        panes = CmuxCli.cmux_json("tree" , "--workspace" , self.ref)['windows'][-1]['workspaces'][-1]['panes']
        surface = [surface for pane in panes for surface in pane['surfaces'] if surface['ref'] == ret['surface_ref']][-1]
        return self.add_child(surface)

class CmuxSurface:
    def __init__(self , ref: str , focused: bool , title: str , workspace: CmuxWorkspace , window: CmuxWindow , **kwargs):
        self.ref = ref
        self.focused = focused
        self.title = title
        self.workspace = workspace
        self.window = window
        self.kwargs = kwargs

    def __repr__(self):
        return f"Surface(window={self.window.id}, workspace={self.workspace.ref}, ref={self.ref}, focused={self.focused}, title={self.title})"

    def save_workspace(self , title : str | None = None):
        if title:
            CmuxTree.SavedWorkspace[title] = self.workspace.ref
        return self

    def focus(self , window : bool = False , surface : bool = True):
        if window:
            self.window.focus()
        if surface:
            self.workspace.select()
            CmuxCli.cmux('focus-panel', '--panel', self.ref , '--workspace', self.workspace.ref)
        return self

    def cwd(self, cwd: str | None = None):
        if cwd:
            CmuxCli.cmux('send', '--workspace', self.workspace.ref , '--surface', self.ref, f'cd {shlex.quote(cwd)}\n')
        return self

    def send(self, command: str , * , cwd: str | None = None) :
        self.cwd(cwd)
        CmuxCli.cmux("send", "--workspace", self.workspace.ref , "--surface", self.ref, f'{command}\n')
        return self

    def rename(self , title: str | None = None , where: Literal["workspace", "surface", "all"] = "surface"):
        if not title:
            return self
        match where:
            case "workspace":
                CmuxCli.cmux('rename-workspace', '--workspace', self.workspace.ref, title)
            case "surface":
                CmuxCli.cmux('rename-tab', '--surface', self.ref, '--workspace', self.workspace.ref, title)
            case "all":
                CmuxCli.cmux('rename-workspace', '--workspace', self.workspace.ref, title)
                CmuxCli.cmux('rename-tab', '--surface', self.ref, '--workspace', self.workspace.ref, title)
        return self

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

def run_in_new_window(command: str , * , cwd: str | None = None, title : str | None = None, 
                      as_workspace: str | None = None):
    surface = CmuxTree.new_window(title = title , as_workspace=as_workspace)
    surface.send(command , cwd=cwd)
def run_in_new_workspace(command: str , * , cwd: str | None = None, title : str | None = None , 
                         as_workspace: str | None = None):
    surface = CmuxTree.new_workspace(title = title, as_workspace=as_workspace)
    surface.send(command , cwd=cwd)

def run_in_new_surface(command: str , * , cwd: str | None = None, title : str | None = None , 
                       from_workspace: str | None = None):
    surface = CmuxTree.new_surface(title=title, from_workspace=from_workspace)
    surface.send(command , cwd=cwd)

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
