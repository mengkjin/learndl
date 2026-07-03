"""Magic stdin commands for interactive CLI sessions."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import questionary
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from src.proj.log import Logger
from src.proj.util.cli.help_context import print_ask_help
from src.proj.util.cli.questionary_style import CLI_SELECT_STYLE
from src.proj.util.cli.session import ProcessQuit, ProcessReload, ProcessSpawn, ProcessSpawnDown

__all__ = [
    'MagicCommand',
    'MAGIC_COMMANDS',
    'MAGIC_INPUT_CATALOG',
    'MAGIC_INPUT_HINT',
    'MAGIC_MENU_VALUE',
    'MAGIC_EXIT_VALUE',
    'MAGIC_CHOICE_PREFIX',
    'MagicInputResult',
    'magic_autocomplete_tokens',
    'magic_autocomplete_meta',
    'MagicCommandCompleter',
    'magic_questionary_choices',
    'append_magic_menu_choice',
    'is_magic_menu_value',
    'is_magic_exit_value',
    'is_magic_choice_value',
    'resolve_magic_choice',
    'run_magic_submenu',
    'print_magic_input_help',
    'resolve_magic_input',
    'set_magic_spawn_handler',
]

MagicInputResult = Literal['hint', 'none', 'not_magic']
MagicAction = Literal['ls_magic', 'help', 'restart', 'spawn', 'spawn_down', 'quit', 'history']

MAGIC_CHOICE_PREFIX = '__magic__:cmd:'
MAGIC_MENU_VALUE = '__magic__:menu'
MAGIC_EXIT_VALUE = '__magic__:exit'
MAGIC_EXIT_LABEL = '« Back (Esc) »'

_magic_spawn_handler: Callable[[bool], None] | None = None


def set_magic_spawn_handler(handler: Callable[[bool], None] | None) -> None:
    """Register a callback for ``/spawn`` / ``/spawn_down`` (``vertical=True`` for the latter)."""
    global _magic_spawn_handler
    _magic_spawn_handler = handler


@dataclass(frozen=True, slots=True)
class MagicCommand:
    """A typed magic command available in text prompts."""

    token: str
    description: str
    action: MagicAction


MAGIC_COMMANDS: tuple[MagicCommand, ...] = (
    MagicCommand('/restart', 'Exec-restart in this terminal to pick up new code', 'restart'),
    MagicCommand('/spawn', 'Open a copy of this DirectCall in a new terminal pane', 'spawn'),
    MagicCommand('/spawn_down', 'Open a copy of this DirectCall in a new pane below', 'spawn_down'),
    MagicCommand('/quit', 'Exit the current process', 'quit'),
    MagicCommand('/ls_magic', 'List all magic commands and re-prompt', 'ls_magic'),
    MagicCommand('/help', 'Show detailed help for the current prompt and re-prompt', 'help'),
    MagicCommand('/history', 'Pick from the 10 most recent inputs for this prompt', 'history'),
)

MAGIC_INPUT_CATALOG: tuple[tuple[str, str], ...] = tuple(
    (cmd.token, cmd.description) for cmd in MAGIC_COMMANDS
)
MAGIC_INPUT_HINT = ' (q to go-back, / for magic calls)'


def magic_autocomplete_tokens() -> list[str]:
    """Tokens offered for autocomplete when typing magic commands."""
    return [command.token for command in MAGIC_COMMANDS]


def magic_autocomplete_meta() -> dict[str, str]:
    """Description metadata shown beside autocomplete suggestions."""
    return {command.token: command.description for command in MAGIC_COMMANDS}


class MagicCommandCompleter(Completer):
    """Magic-command completer that replaces the typed prefix (fixes WordCompleter overlap)."""

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        prefix = document.text_before_cursor
        prefix_lower = prefix.lower()
        meta = magic_autocomplete_meta()
        for token in magic_autocomplete_tokens():
            token_lower = token.lower()
            if prefix_lower:
                if not token_lower.startswith(prefix_lower):
                    continue
            yield Completion(
                token,
                start_position=-len(prefix),
                display_meta=meta.get(token, ''),
            )


def magic_questionary_choices() -> list[questionary.Choice]:
    """Build questionary choices for the magic command submenu."""
    return [
        questionary.Choice(MAGIC_EXIT_LABEL, value=MAGIC_EXIT_VALUE),
        questionary.Separator(),
        *[
            questionary.Choice(f'{command.token} — {command.description}', value=f'{MAGIC_CHOICE_PREFIX}{command.token}')
            for command in MAGIC_COMMANDS
        ],
    ]


def append_magic_menu_choice(choices: list[questionary.Choice]) -> list[Any]:
    """Append a single entry that opens the magic command submenu."""
    return [
        *choices,
        questionary.Separator(),
        questionary.Choice('Magic commands...', value=MAGIC_MENU_VALUE),
    ]


def is_magic_menu_value(value: Any) -> bool:
    return value == MAGIC_MENU_VALUE


def is_magic_exit_value(value: Any) -> bool:
    return value == MAGIC_EXIT_VALUE


def is_magic_choice_value(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(MAGIC_CHOICE_PREFIX)


def resolve_magic_choice(value: Any) -> MagicInputResult | str:
    """Handle a magic submenu selection; return ``'not_magic'`` for normal values."""
    if not is_magic_choice_value(value):
        return 'not_magic'
    token = value.removeprefix(MAGIC_CHOICE_PREFIX)
    return resolve_magic_input(token)


def run_magic_submenu() -> Literal['hint', 'cancelled']:
    """Second-level picker for magic commands."""
    while True:
        value = questionary.select(
            'Magic commands',
            choices=magic_questionary_choices(),
            style=CLI_SELECT_STYLE,
        ).ask()
        if value is None or is_magic_exit_value(value):
            return 'cancelled'
        if resolve_magic_choice(value) == 'hint':
            return 'hint'


def print_magic_input_help() -> None:
    """Print descriptions for every registered magic command."""
    Logger.stdout('Available magic inputs:', color='lightcyan')
    for command, description in MAGIC_INPUT_CATALOG:
        Logger.stdout(f'{command}: {description}', indent=1)


def resolve_magic_input(raw: str) -> MagicInputResult | str:
    """Handle magic stdin commands in text prompts.

    Returns ``'hint'`` to re-prompt (including after ``/history`` prefill),
    ``'none'`` when *raw* is not a magic command.
    Raises :class:`ProcessReload`, :class:`ProcessQuit`, or spawn exceptions when applicable.
    """
    token = raw.strip().lower()
    for command in MAGIC_COMMANDS:
        if token != command.token:
            continue
        if command.action == 'history':
            from src.proj.util.cli.input_history import (
                get_active_input_history,
                pick_history_entry,
                stage_input_prefill,
            )

            history = get_active_input_history()
            if history is None:
                Logger.note('/history is only available during path or expression prompts.')
                return 'hint'
            picked = pick_history_entry(history)
            if picked is None:
                return 'hint'
            stage_input_prefill(picked)
            return 'hint'
        if command.action == 'ls_magic':
            print_magic_input_help()
            return 'hint'
        if command.action == 'help':
            print_ask_help()
            return 'hint'
        if command.action == 'restart':
            raise ProcessReload(f'manual {token}')
        if command.action == 'spawn':
            if _magic_spawn_handler is not None:
                _magic_spawn_handler(False)
                return 'hint'
            raise ProcessSpawn(f'manual {token}')
        if command.action == 'spawn_down':
            if _magic_spawn_handler is not None:
                _magic_spawn_handler(True)
                return 'hint'
            raise ProcessSpawnDown(f'manual {token}')
        if command.action == 'quit':
            raise ProcessQuit(f'manual {token}')
    return 'none'
