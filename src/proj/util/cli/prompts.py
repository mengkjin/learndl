"""questionary-backed prompt helpers for terminal interaction."""
from __future__ import annotations

from typing import Any, TypeVar

import questionary
from prompt_toolkit.key_binding import KeyBindings, KeyBindingsBase, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import CompleteStyle

from src.proj.util.cli.input_history import (
    active_input_history,
    commit_input_history,
    expression_input_history,
    path_input_history,
    take_input_prefill,
)
from src.proj.util.cli.magic import (
    MagicInputResult,
    MagicCommandCompleter,
    append_magic_menu_choice,
    is_magic_choice_value,
    is_magic_menu_value,
    resolve_magic_choice,
    resolve_magic_input,
    run_magic_submenu,
)
from src.proj.util.cli.questionary_style import CLI_INPUT_STYLE, CLI_SELECT_STYLE

__all__ = [
    'prompt_text',
    'prompt_project_path',
    'prompt_expression',
    'prompt_select',
    'prompt_checkbox',
    'prompt_confirm',
    'format_input_prompt',
]

T = TypeVar('T')


def format_input_prompt(message: str, *, allow_magic: bool = True) -> str:
    """Append a short go-back / magic hint to *message*."""
    if allow_magic:
        return f'{message} (q to go-back, / for magic calls)'
    return f'{message} (q to go-back)'


def _handle_magic_menu_pick() -> bool:
    """Run magic submenu; return True when the outer prompt should re-display."""
    submenu_result = run_magic_submenu()
    return submenu_result in {'hint', 'cancelled'}


def _resolve_magic_text_input(value: str) -> MagicInputResult | str:
    """Resolve magic commands; return a concrete string when a command supplies one."""
    return resolve_magic_input(value)


def _build_accept_completion_key_bindings() -> KeyBindings:
    """Enter accepts the highlighted completion when the menu is open, else submits."""
    bindings = KeyBindings()

    @bindings.add(Keys.ControlM, eager=True)
    def accept_completion_or_submit(event: KeyPressEvent) -> None:
        buff = event.current_buffer
        if buff.complete_state is not None:
            completion = buff.complete_state.current_completion
            if completion is not None:
                buff.apply_completion(completion)
            buff.complete_state = None
        else:
            buff.append_to_history()
            event.app.exit(result=buff.text)

    return bindings


def _build_navigation_key_bindings() -> KeyBindings:
    """Up/Down/Tab navigate the path or magic completion menu."""
    bindings = KeyBindings()

    @bindings.add(Keys.Up, eager=True)
    def arrow_up(event: KeyPressEvent) -> None:
        buff = event.current_buffer
        if buff.complete_state is None:
            buff.start_completion(select_first=False)
        if buff.complete_state is not None:
            buff.complete_previous()

    @bindings.add(Keys.Down, eager=True)
    def arrow_down(event: KeyPressEvent) -> None:
        buff = event.current_buffer
        if buff.complete_state is None:
            buff.start_completion(select_first=False)
        if buff.complete_state is not None:
            buff.complete_next()

    @bindings.add(Keys.Tab, eager=True)
    def tab_complete(event: KeyPressEvent) -> None:
        buff = event.current_buffer
        if buff.complete_state is not None:
            buff.complete_next()
        else:
            buff.start_completion(select_first=True)

    return bindings


def _merged_prompt_key_bindings(extra: KeyBindings | None = None) -> KeyBindingsBase:
    """Merge default prompt_toolkit bindings with completion, navigation, and *extra*."""
    extras: list[KeyBindings] = [_build_accept_completion_key_bindings()]
    if extra is not None:
        extras.append(extra)
    extras.append(_build_navigation_key_bindings())
    return merge_key_bindings([load_key_bindings(), *extras])


def _questionary_autocomplete(
    message: str,
    *,
    completer,
    history=None,
    complete_style: CompleteStyle = CompleteStyle.COLUMN,
    key_bindings: KeyBindings | None = None,
    complete_while_typing: bool = False,
) -> str | None:
    prefill = take_input_prefill() or ''
    return questionary.autocomplete(
        message,
        choices=[],
        completer=completer,
        default=prefill,
        complete_while_typing=complete_while_typing,
        complete_style=complete_style,
        history=history,
        enable_history_search=False,
        key_bindings=_merged_prompt_key_bindings(key_bindings),
        style=CLI_INPUT_STYLE,
    ).ask()


def prompt_text(message: str, *, allow_magic: bool = True) -> str | None:
    """Read a text value with optional magic autocomplete. Returns ``None`` on cancel."""
    prompt = format_input_prompt(message, allow_magic=allow_magic)
    while True:
        if allow_magic:
            value = _questionary_autocomplete(
                prompt,
                completer=MagicCommandCompleter(),
            )
        else:
            value = questionary.text(prompt, style=CLI_INPUT_STYLE).ask()
        if value is None:
            return None
        if allow_magic:
            magic_result = _resolve_magic_text_input(value)
            if magic_result == 'hint':
                continue
            if magic_result != 'none':
                return magic_result
        return value


def prompt_project_path(message: str, *, allow_magic: bool = True) -> str | None:
    """Read a project-relative path via questionary autocomplete and magic."""
    from src.proj.util.cli.project_path import (
        MagicOrProjectPathCompleter,
        ProjectPathCompleter,
        build_path_prompt_key_bindings,
    )

    prompt = format_input_prompt(message, allow_magic=allow_magic)
    completer = MagicOrProjectPathCompleter() if allow_magic else ProjectPathCompleter()
    with active_input_history(path_input_history):
        while True:
            value = _questionary_autocomplete(
                prompt,
                completer=completer,
                history=path_input_history,
                complete_style=CompleteStyle.COLUMN,
                key_bindings=build_path_prompt_key_bindings(),
                complete_while_typing=True,
            )
            if value is None:
                return None
            if allow_magic:
                magic_result = _resolve_magic_text_input(value)
                if magic_result == 'hint':
                    continue
                if magic_result != 'none':
                    return magic_result
            commit_input_history(path_input_history, value)
            return value


def prompt_expression(message: str, *, allow_magic: bool = True) -> str | None:
    """Read an expression via questionary autocomplete with magic."""
    prompt = format_input_prompt(message, allow_magic=allow_magic)
    completer = MagicCommandCompleter() if allow_magic else None
    with active_input_history(expression_input_history):
        while True:
            value = _questionary_autocomplete(
                prompt,
                completer=completer,
                history=expression_input_history,
            )
            if value is None:
                return None
            if allow_magic:
                magic_result = _resolve_magic_text_input(value)
                if magic_result == 'hint':
                    continue
                if magic_result != 'none':
                    return magic_result
            commit_input_history(expression_input_history, value)
            return value


def prompt_select(message: str, choices: list[questionary.Choice], *, allow_magic: bool = True) -> Any | None:
    """Single-select prompt. Returns the choice ``value`` or ``None`` on cancel."""
    while True:
        menu_choices = append_magic_menu_choice(choices) if allow_magic else choices
        value = questionary.select(
            message,
            choices=menu_choices,
            style=CLI_SELECT_STYLE,
        ).ask()
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
        selected = questionary.checkbox(
            message,
            choices=menu_choices,
            style=CLI_SELECT_STYLE,
        ).ask()
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
        return questionary.confirm(message, default=default, style=CLI_SELECT_STYLE).ask()
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
            style=CLI_SELECT_STYLE,
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
