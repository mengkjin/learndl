"""Shared questionary styles for CLI prompts."""

from __future__ import annotations

from typing import cast

from prompt_toolkit.styles import Style
from questionary import Style as QuestionaryStyle
from questionary.styles import merge_styles_default

__all__ = [
    'CLI_INPUT_STYLE',
    'CLI_QUESTIONARY_STYLE',
    'CLI_SELECT_STYLE',
]

_ANSWER_FACE = 'fg:#FF9D00 bold'

# Typed input stays orange; completion popup keeps default readable colors (not orange).
CLI_INPUT_STYLE = cast(
    Style,
    merge_styles_default([
        QuestionaryStyle([
            ('answer', _ANSWER_FACE),
            ('completion-menu.completion', ''),
            ('completion-menu.completion.current', 'bg:#666666 bold'),
            ('selected', ''),
        ]),
    ]),
)

# Select / checkbox / confirm menus: orange pointer and option text.
CLI_SELECT_STYLE = cast(
    Style,
    merge_styles_default([
        QuestionaryStyle([
            ('pointer', _ANSWER_FACE),
            ('highlighted', _ANSWER_FACE),
            ('selected', _ANSWER_FACE),
            ('text', _ANSWER_FACE),
        ]),
    ]),
)

# Backward-compatible alias for menu prompts.
CLI_QUESTIONARY_STYLE = CLI_SELECT_STYLE
