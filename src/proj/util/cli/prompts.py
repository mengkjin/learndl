"""questionary-backed prompt helpers for terminal interaction."""
from __future__ import annotations

from typing import Any, TypeVar

import questionary

from src.proj.util.cli.magic import (
    MAGIC_INPUT_HINT,
    append_magic_menu_choice,
    is_magic_choice_value,
    is_magic_menu_value,
    magic_autocomplete_meta,
    magic_autocomplete_tokens,
    resolve_magic_choice,
    resolve_magic_input,
    run_magic_submenu,
)

__all__ = [
    'prompt_text',
    'prompt_select',
    'prompt_checkbox',
    'prompt_confirm',
]

T = TypeVar('T')


def _handle_magic_menu_pick() -> bool:
    """Run magic submenu; return True when the outer prompt should re-display."""
    submenu_result = run_magic_submenu()
    return submenu_result in {'hint', 'cancelled'}


def prompt_text(message: str, *, allow_magic: bool = True) -> str | None:
    """Read a text value with optional magic autocomplete. Returns ``None`` on cancel."""
    prompt = f'{message}{MAGIC_INPUT_HINT if allow_magic else ""}'
    while True:
        if allow_magic:
            value = questionary.autocomplete(
                prompt,
                choices=magic_autocomplete_tokens(),
                meta_information=magic_autocomplete_meta(),
                match_middle=True,
            ).ask()
        else:
            value = questionary.text(prompt).ask()
        if value is None:
            return None
        if allow_magic and resolve_magic_input(value) == 'hint':
            continue
        return value


def prompt_select(message: str, choices: list[questionary.Choice], *, allow_magic: bool = True) -> Any | None:
    """Single-select prompt. Returns the choice ``value`` or ``None`` on cancel."""
    while True:
        menu_choices = append_magic_menu_choice(choices) if allow_magic else choices
        value = questionary.select(message, choices=menu_choices).ask()
        if value is None:
            return None
        if allow_magic and is_magic_menu_value(value):
            if _handle_magic_menu_pick():
                continue
            return None
        if allow_magic:
            magic_result = resolve_magic_choice(value)
            if magic_result == 'hint':
                continue
        return value


def prompt_checkbox(message: str, choices: list[questionary.Choice], *, allow_magic: bool = True) -> list[Any] | None:
    """Multi-select prompt. Returns selected values or ``None`` on cancel."""
    while True:
        menu_choices = append_magic_menu_choice(choices) if allow_magic else choices
        selected = questionary.checkbox(message, choices=menu_choices).ask()
        if selected is None:
            return None
        if allow_magic and any(is_magic_menu_value(value) for value in selected):
            if _handle_magic_menu_pick():
                continue
            return None
        if allow_magic:
            magic_values = [value for value in selected if is_magic_choice_value(value)]
            normal_values = [value for value in selected if not is_magic_choice_value(value)]
            for magic_value in magic_values:
                if resolve_magic_choice(magic_value) == 'hint':
                    break
            else:
                return normal_values
            continue
        return selected


def prompt_confirm(message: str, *, default: bool = False, allow_magic: bool = True) -> bool | None:
    """Yes/no confirmation. Returns ``None`` on cancel."""
    if not allow_magic:
        return questionary.confirm(message, default=default).ask()
    confirm_choices = [
        questionary.Choice('Yes', value=True),
        questionary.Choice('No', value=False),
    ]
    while True:
        menu_choices = append_magic_menu_choice(confirm_choices)
        value = questionary.select(
            message,
            choices=menu_choices,
            default=confirm_choices[0 if default else 1],
        ).ask()
        if value is None:
            return None
        if is_magic_menu_value(value):
            if _handle_magic_menu_pick():
                continue
            return None
        if isinstance(value, bool):
            return value
        if resolve_magic_choice(value) == 'hint':
            continue
