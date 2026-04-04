from src.proj.env import MACHINE
_preference = MACHINE.configs("preference" , "shell_opener")

PAUSE_WHEN_DONE : bool = _preference["PAUSE_WHEN_DONE"]

DISCOVER_WAIT_TIMEOUT : float = _preference["DISCOVER_WAIT_TIMEOUT"]
DISCOVER_WAIT_INTERVAL : float = _preference["DISCOVER_WAIT_INTERVAL"]

MACOS_OPTIONS : list[str] = _preference["MACOS_OPTIONS"]
MACOS_TERMINAL_PROFILE_NAME : str = _preference["MACOS_TERMINAL_PROFILE_NAME"]
MACOS_CMUX_NEW : str = _preference["MACOS_CMUX_NEW"]
MACOS_CMUX_COLD_START_DEADLINE : float = _preference["MACOS_CMUX_COLD_START_DEADLINE"]
MACOS_CMUX_PING_INTERVAL : float = _preference["MACOS_CMUX_PING_INTERVAL"]
MACOS_CMUX_CMD_TIMEOUT : float = _preference["MACOS_CMUX_CMD_TIMEOUT"]

LINUX_OPTIONS : list[str] = _preference["LINUX_OPTIONS"]
LINUX_GNOME_NEW : str = _preference["LINUX_GNOME_NEW"]

WINDOWS_OPTIONS : list[str] = _preference["WINDOWS_OPTIONS"]