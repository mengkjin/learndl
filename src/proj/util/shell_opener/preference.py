from typing import Literal
from src.proj.env import CONST

LiteralOption = Literal["cmux", "ghostty", "terminal.app", "gnome", "cmd"]
LiteralNewOn = Literal["window", "workspace", "tab"]

PAUSE_WHEN_DONE : bool = CONST.Pref.get("shell_opener" , "pause_when_done" , True)

DISCOVER_WAIT_TIMEOUT : float = CONST.Pref.get("shell_opener" , "discover_wait_timeout" , 15.0)
DISCOVER_WAIT_INTERVAL : float = CONST.Pref.get("shell_opener" , "discover_wait_interval" , 0.25)

MACOS_OPTIONS : list[LiteralOption] = CONST.Pref.get("shell_opener" , "macos_options" , ["cmux", "ghostty", "terminal.app"])
MACOS_TERMINAL_PROFILE_NAME : str = CONST.Pref.get("shell_opener" , "macos_terminal_profile_name" , "Basic")
MACOS_CMUX_NEW : LiteralNewOn = CONST.Pref.get("shell_opener" , "macos_cmux_new" , "tab")
MACOS_CMUX_COLD_START_DEADLINE : float = CONST.Pref.get("shell_opener" , "macos_cmux_cold_start_deadline" , 15.0)
MACOS_CMUX_PING_INTERVAL : float = CONST.Pref.get("shell_opener" , "macos_cmux_ping_interval" , 0.25)
MACOS_CMUX_CMD_TIMEOUT : float = CONST.Pref.get("shell_opener" , "macos_cmux_cmd_timeout" , 120.0)

LINUX_OPTIONS : list[LiteralOption] = CONST.Pref.get("shell_opener" , "linux_options" , ["gnome"])
LINUX_GNOME_NEW : LiteralNewOn = CONST.Pref.get("shell_opener" , "linux_gnome_new" , "tab")

WINDOWS_OPTIONS : list[LiteralOption] = CONST.Pref.get("shell_opener" , "windows_options" , ["cmd"])
WINDOWS_CMD_NEW : LiteralNewOn = CONST.Pref.get("shell_opener" , "windows_cmd_new" , "tab")
