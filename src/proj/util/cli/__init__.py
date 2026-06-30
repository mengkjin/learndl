"""Terminal interaction: prompts, magic commands, and session reload helpers."""
from __future__ import annotations

from src.proj.util.cli.ask import AskFlag, AskFlagType, AskFor, LoopFlag
from src.proj.util.cli.magic import (
    MAGIC_INPUT_CATALOG,
    MAGIC_INPUT_HINT,
    MagicCommand,
    resolve_magic_input,
)
from src.proj.util.cli.session import (
    GitHeadWatcher,
    ProcessQuit,
    ProcessReload,
    ProcessSpawn,
    ProcessSpawnDown,
    build_direct_call_script,
    build_exec_argv,
    can_exec_restart,
    git_head,
)
from src.proj.util.cli.help_context import AskHelpContext, print_ask_help, set_ask_help_context
from src.proj.util.cli.script_params import prompt_script_kwargs, run_script_interactive
from src.proj.util.cli.script_session import as_script_main
from src.proj.util.script.param_schema import ScriptParamSchema

__all__ = [
    'AskFor',
    'AskFlag',
    'AskFlagType',
    'LoopFlag',
    'MagicCommand',
    'MAGIC_INPUT_CATALOG',
    'MAGIC_INPUT_HINT',
    'ProcessReload',
    'ProcessSpawn',
    'ProcessSpawnDown',
    'ProcessQuit',
    'AskHelpContext',
    'print_ask_help',
    'set_ask_help_context',
    'GitHeadWatcher',
    'git_head',
    'resolve_magic_input',
    'can_exec_restart',
    'build_direct_call_script',
    'build_exec_argv',
    'prompt_script_kwargs',
    'run_script_interactive',
    'as_script_main',
    'ScriptParamSchema',
]
