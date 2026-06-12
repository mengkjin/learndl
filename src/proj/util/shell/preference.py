"""User-configurable defaults for shell, loaded from ``CONST.Pref`` under the ``shell`` section."""
from __future__ import annotations
from src.proj.env import Const

__all__ = [
    'DONE_ACTION' , 
    'DISCOVER_WAIT_TIMEOUT' , 'DISCOVER_WAIT_INTERVAL' , 
    
    'MACOS_OPTIONS' , 
    'MACOS_TERMINAL_PROFILE_NAME' , 
    'MACOS_CMUX_NEW' , 'MACOS_CMUX_COLD_START_DEADLINE' , 'MACOS_CMUX_PING_INTERVAL' , 'MACOS_CMUX_CMD_TIMEOUT' , 'MACOS_WEZTERM_NEW' , 

    'LINUX_OPTIONS' , 
    'LINUX_WEZTERM_NEW' , 
    'LINUX_GNOME_NEW' , 

    'WINDOWS_OPTIONS' , 
    'WINDOWS_CMD_NEW' , 
    'WINDOWS_WEZTERM_NEW'
]

_pref = Const.Pref.shell

DONE_ACTION = _pref.get("done_action" , 'pause')

DISCOVER_WAIT_TIMEOUT : float = _pref.get("discover_wait_timeout" , 15.0)
DISCOVER_WAIT_INTERVAL : float = _pref.get("discover_wait_interval" , 0.25)

MACOS_OPTIONS : list[str] = _pref.get("macos_options" , ["cmux", "wezterm", "ghostty", "terminal.app"])
MACOS_TERMINAL_PROFILE_NAME : str = _pref.get("macos_terminal_profile_name" , "Basic")
MACOS_CMUX_NEW : str = _pref.get("macos_cmux_new" , "tab")
MACOS_CMUX_COLD_START_DEADLINE : float = _pref.get("macos_cmux_cold_start_deadline" , 15.0)
MACOS_CMUX_PING_INTERVAL : float = _pref.get("macos_cmux_ping_interval" , 0.25)
MACOS_CMUX_CMD_TIMEOUT : float = _pref.get("macos_cmux_cmd_timeout" , 120.0)
MACOS_WEZTERM_NEW: str = _pref.get("macos_wezterm_new", "tab")

LINUX_OPTIONS : list[str] = _pref.get("linux_options" , ["wezterm", "gnome"])
LINUX_WEZTERM_NEW: str = _pref.get("linux_wezterm_new", "tab")
LINUX_GNOME_NEW : str = _pref.get("linux_gnome_new" , "tab")

WINDOWS_OPTIONS : list[str] = _pref.get("windows_options" , ["wezterm", "cmd"])
WINDOWS_CMD_NEW : str = _pref.get("windows_cmd_new" , "tab")
WINDOWS_WEZTERM_NEW: str = _pref.get("windows_wezterm_new", "tab")
