"""macOS Terminal.app (AppleScript)."""

from __future__ import annotations

import shlex

from ...util.process import popen_detached
from ...util.basic import BasicOpener
from .verify import TerminalAppVerifier

def _applescript_escape(s: str) -> str:
    """Escape backslashes and double-quotes for embedding in an AppleScript string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')

class TerminalAppOpener(BasicOpener):
    """Open commands in a new Terminal.app window via AppleScript (``osascript``)."""

    def available(self) -> bool:
        """Return True if ``osascript`` is available (Terminal.app automation is enabled)."""
        return TerminalAppVerifier.available()

    def run(self, command: str, * , cwd: str | None = None, **kwargs) -> None:
        """Open a new Terminal.app tab/window running ``cd`` + ``command``."""
        assert self._available , f"{self.__class__.__name__} is not available"
        if cwd:
            command = f"cd {shlex.quote(cwd)} && {command}"
        apple_script_cmd = _applescript_escape(command)
        script = f'''
    tell application "Terminal"
        -- Create a new terminal window
        set new_window to do script ""
        set current settings of new_window to settings set "Basic"
        do script "{apple_script_cmd}" in new_window
        -- Bring the main application window to the front
        activate
    end tell
    '''
        # script = f'tell application "Terminal" to do script "{_applescript_escape(inner)}"'
        popen_detached(["osascript", "-e", script])