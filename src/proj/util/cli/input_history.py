"""In-process input history for interactive CLI prompts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

import questionary
from prompt_toolkit.history import InMemoryHistory

from src.proj import Logger
from src.proj.util.cli.questionary_style import CLI_SELECT_STYLE

__all__ = [
    'HISTORY_PICK_LIMIT',
    'active_input_history',
    'commit_input_history',
    'expression_input_history',
    'get_active_input_history',
    'path_input_history',
    'pick_history_entry',
    'stage_input_prefill',
    'take_input_prefill',
]

HISTORY_PICK_LIMIT = 10

path_input_history = InMemoryHistory()
expression_input_history = InMemoryHistory()

_active_history: ContextVar[InMemoryHistory | None] = ContextVar('active_input_history', default=None)
_pending_prefill: ContextVar[str | None] = ContextVar('pending_input_prefill', default=None)


@contextmanager
def active_input_history(history: InMemoryHistory) -> Iterator[None]:
    """Bind *history* for ``/history`` magic during a prompt loop."""
    token = _active_history.set(history)
    try:
        yield
    finally:
        _active_history.reset(token)


def get_active_input_history() -> InMemoryHistory | None:
    return _active_history.get()


def stage_input_prefill(value: str) -> None:
    """Pre-fill the next autocomplete prompt; user must press Enter to submit."""
    _pending_prefill.set(value)


def take_input_prefill() -> str | None:
    """Return and clear a staged prefill value for the next prompt."""
    value = _pending_prefill.get()
    if value is not None:
        _pending_prefill.set(None)
    return value


def commit_input_history(history: InMemoryHistory, value: str) -> None:
    """Persist a submitted prompt value for ``/history`` (skips magic commands)."""
    text = value.strip()
    if not text or text.startswith('/'):
        return
    prior = history.get_strings()
    if prior and prior[-1] == text:
        return
    history.append_string(text)


def pick_history_entry(
    history: InMemoryHistory,
    *,
    title: str = 'Recent inputs (newest first)',
) -> str | None:
    """Show up to :data:`HISTORY_PICK_LIMIT` history lines in a select menu."""
    items = list(history.get_strings())[-HISTORY_PICK_LIMIT:][::-1]
    if not items:
        Logger.note('No history entries yet.')
        return None
    choices = [questionary.Choice(item, value=item) for item in items]
    return questionary.select(
        title,
        choices=choices,
        style=CLI_SELECT_STYLE,
        instruction='(Use arrow keys)',
    ).ask()
