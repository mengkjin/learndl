"""Utility helpers re-exported for the shell package (pause, commands, discovery, process)."""

from .pausing import compose_with_done_action
from .commands import format_python_command , to_shell_string , guess_command_title
from .discovery import ProcessDiscovery
from .basic import BasicOpener
from .argparse import argparse_dict
from . import process

__all__ = [
    "compose_with_done_action", "format_python_command", "to_shell_string", 
    "ProcessDiscovery", "process", "guess_command_title", "BasicOpener", "argparse_dict"
]