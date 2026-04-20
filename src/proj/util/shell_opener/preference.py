"""User-configurable defaults for shell_opener, loaded from ``CONST.Pref`` under the ``shell_opener`` section."""

from typing import Literal
from src.proj.env import Const

LiteralOption = Literal["cmux", "ghostty", "terminal.app", "wezterm", "gnome", "cmd"]
LiteralNewOn = Literal["window", "workspace", "tab"]

_pref = Const.Pref.shell_opener

PAUSE_WHEN_DONE : bool = _pref.get("pause_when_done" , True)

DISCOVER_WAIT_TIMEOUT : float = _pref.get("discover_wait_timeout" , 15.0)
DISCOVER_WAIT_INTERVAL : float = _pref.get("discover_wait_interval" , 0.25)

MACOS_OPTIONS : list[LiteralOption] = _pref.get("macos_options" , ["cmux", "wezterm", "ghostty", "terminal.app"])
MACOS_TERMINAL_PROFILE_NAME : str = _pref.get("macos_terminal_profile_name" , "Basic")
MACOS_CMUX_NEW : LiteralNewOn = _pref.get("macos_cmux_new" , "tab")
MACOS_CMUX_COLD_START_DEADLINE : float = _pref.get("macos_cmux_cold_start_deadline" , 15.0)
MACOS_CMUX_PING_INTERVAL : float = _pref.get("macos_cmux_ping_interval" , 0.25)
MACOS_CMUX_CMD_TIMEOUT : float = _pref.get("macos_cmux_cmd_timeout" , 120.0)
MACOS_WEZTERM_NEW: LiteralNewOn = _pref.get("macos_wezterm_new", "tab")

LINUX_OPTIONS : list[LiteralOption] = _pref.get("linux_options" , ["wezterm", "gnome"])
LINUX_WEZTERM_NEW: LiteralNewOn = _pref.get("linux_wezterm_new", "tab")
LINUX_GNOME_NEW : LiteralNewOn = _pref.get("linux_gnome_new" , "tab")

WINDOWS_OPTIONS : list[LiteralOption] = _pref.get("windows_options" , ["wezterm", "cmd"])
WINDOWS_CMD_NEW : LiteralNewOn = _pref.get("windows_cmd_new" , "tab")
WINDOWS_WEZTERM_NEW: LiteralNewOn = _pref.get("windows_wezterm_new", "tab")
