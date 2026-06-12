"""Utility helpers re-exported for the shell package (pause, commands, discovery, process)."""

from .pausing import compose_with_done_action , DoneActionType
from .commands import (
    format_python_command,
    to_shell_string,
    guess_command_title,
    wrap_cmd_exe_line,
    prepare_cmd_k_line,
)
from .discovery import ProcessDiscovery
from .basic import BasicOpener
from .argparse import argparse_dict
from . import process

__all__ = [
    "compose_with_done_action" , "DoneActionType", "format_python_command", "to_shell_string", 
    "ProcessDiscovery", "process", "guess_command_title", "wrap_cmd_exe_line", "prepare_cmd_k_line",
    "BasicOpener", "argparse_dict"
]