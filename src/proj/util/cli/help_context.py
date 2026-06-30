"""Per-prompt help context for the ``/help`` magic command."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.proj.log import Logger

__all__ = [
    'AskHelpContext',
    'build_option_help_map',
    'set_ask_help_context',
    'print_ask_help',
]

_ask_help_context: AskHelpContext | None = None


@dataclass
class AskHelpContext:
    """Describe the current AskFor prompt for ``/help`` output."""

    prompt_title: str = ''
    description: str = ''
    options: list[Any] = field(default_factory=list)
    option_help: dict[str, str] = field(default_factory=dict)
    extra_lines: tuple[str, ...] = ()


def build_option_help_map(
    options: Sequence[Any],
    option_help: Mapping[Any, str] | Sequence[str] | None,
) -> dict[str, str]:
    """Normalize per-option help as ``{str(option): detail}``."""
    if option_help is None:
        return {}
    if isinstance(option_help, Mapping):
        return {str(key): value for key, value in option_help.items()}
    return {
        str(option): detail
        for option, detail in zip(options, option_help, strict=False)
        if detail
    }


def set_ask_help_context(context: AskHelpContext | None) -> None:
    """Register help shown by ``/help`` for the active prompt."""
    global _ask_help_context
    _ask_help_context = context


def print_ask_help() -> bool:
    """Print registered prompt help. Returns False when no context is set."""
    if _ask_help_context is None:
        Logger.note('No help context for this prompt. Use AskFor(..., help_description=...) when calling.')
        return False

    ctx = _ask_help_context
    Logger.stdout('Prompt help:', color='lightcyan')
    if ctx.prompt_title:
        Logger.stdout(ctx.prompt_title, indent=1, color='lightpurple')
    if ctx.description:
        Logger.stdout(ctx.description, indent=1)
    for line in ctx.extra_lines:
        Logger.stdout(line, indent=1)
    if ctx.options:
        Logger.stdout('Options:', indent=1)
        for index, option in enumerate(ctx.options, start=1):
            detail = ctx.option_help.get(str(option), '')
            label = f'{index:02d}. {option}'
            if detail:
                label = f'{label} — {detail}'
            Logger.stdout(label, indent=2)
    return True
