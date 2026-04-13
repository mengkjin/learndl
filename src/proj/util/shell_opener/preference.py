"""User-configurable defaults for shell_opener, loaded from ``CONST.Pref`` under the ``shell_opener`` section."""

from typing import Literal , Any
from src.proj.env import CONST

def perf_getter(key: str , default: Any) -> Any:
    """Read a ``shell_opener`` preference key, returning ``default`` if absent."""
    return CONST.Pref.get('shell_opener', key, default)

LiteralOption = Literal["cmux", "ghostty", "terminal.app", "wezterm", "gnome", "cmd"]
LiteralNewOn = Literal["window", "workspace", "tab"]

PAUSE_WHEN_DONE : bool = perf_getter("pause_when_done" , True)

DISCOVER_WAIT_TIMEOUT : float = perf_getter("discover_wait_timeout" , 15.0)
DISCOVER_WAIT_INTERVAL : float = perf_getter("discover_wait_interval" , 0.25)

MACOS_OPTIONS : list[LiteralOption] = perf_getter("macos_options" , ["cmux", "wezterm", "ghostty", "terminal.app"])
MACOS_TERMINAL_PROFILE_NAME : str = perf_getter("macos_terminal_profile_name" , "Basic")
MACOS_CMUX_NEW : LiteralNewOn = perf_getter("macos_cmux_new" , "tab")
MACOS_CMUX_COLD_START_DEADLINE : float = perf_getter("macos_cmux_cold_start_deadline" , 15.0)
MACOS_CMUX_PING_INTERVAL : float = perf_getter("macos_cmux_ping_interval" , 0.25)
MACOS_CMUX_CMD_TIMEOUT : float = perf_getter("macos_cmux_cmd_timeout" , 120.0)
MACOS_WEZTERM_NEW: LiteralNewOn = perf_getter("macos_wezterm_new", "tab")

LINUX_OPTIONS : list[LiteralOption] = perf_getter("linux_options" , ["wezterm", "gnome"])
LINUX_WEZTERM_NEW: LiteralNewOn = perf_getter("linux_wezterm_new", "tab")
LINUX_GNOME_NEW : LiteralNewOn = perf_getter("linux_gnome_new" , "tab")

WINDOWS_OPTIONS : list[LiteralOption] = perf_getter("windows_options" , ["wezterm", "cmd"])
WINDOWS_CMD_NEW : LiteralNewOn = perf_getter("windows_cmd_new" , "tab")
WINDOWS_WEZTERM_NEW: LiteralNewOn = perf_getter("windows_wezterm_new", "tab")
