from pathlib import Path

import yaml

_base_path_of_terminal = 'learndl'

def get_root_path():
    path = Path(__file__).resolve()
    while path.name != _base_path_of_terminal:
        path = path.parent
    return path

ROOT = get_root_path()

with ROOT.joinpath("configs" , "preference" , "terminal_opener.yaml").open("r", encoding="utf-8") as f:
    preference = yaml.safe_load(f)

PAUSE_WHEN_DONE : bool = preference["PAUSE_WHEN_DONE"]

DISCOVER_WAIT_TIMEOUT : float = preference["DISCOVER_WAIT_TIMEOUT"]
DISCOVER_WAIT_INTERVAL : float = preference["DISCOVER_WAIT_INTERVAL"]

MACOS_OPTIONS : list[str] = preference["MACOS_OPTIONS"]
MACOS_TERMINAL_PROFILE_NAME : str = preference["MACOS_TERMINAL_PROFILE_NAME"]
MACOS_CMUX_NEW : str = preference["MACOS_CMUX_NEW"]
MACOS_CMUX_COLD_START_DEADLINE : float = preference["MACOS_CMUX_COLD_START_DEADLINE"]
MACOS_CMUX_PING_INTERVAL : float = preference["MACOS_CMUX_PING_INTERVAL"]
MACOS_CMUX_CMD_TIMEOUT : float = preference["MACOS_CMUX_CMD_TIMEOUT"]

LINUX_OPTIONS : list[str] = preference["LINUX_OPTIONS"]
LINUX_GNOME_NEW : str = preference["LINUX_GNOME_NEW"]

WINDOWS_OPTIONS : list[str] = preference["WINDOWS_OPTIONS"]