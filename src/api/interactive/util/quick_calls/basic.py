"""
Quick call buttons for the interactive app, can be directly used in any page.
"""
from __future__ import annotations

from abc import abstractmethod , ABCMeta
from typing import Literal , TypeAlias , Any , get_args

from src.proj.util.shell.util import DoneActionType
from src.api.interactive.util.components.operations import ButtonOperation

__all__ = ['QuickCallButton']

EncodedColor : TypeAlias = Literal[
    'red' , 'green' , 'blue' , 'orange' , 'purple' , 
    'gray' , 'yellow' , 'pink' , 'gold' , 'cyan'
]

class QuickCallButtonMeta(ABCMeta):
    """Meta class for QuickCallButton."""
    registry : dict[str, type[QuickCallButton] | Any] = {}
    def __new__(cls , name , bases , dct):
        new_cls = super().__new__(cls , name , bases , dct)
        if name != 'QuickCallButton':
            assert getattr(new_cls, 'done_action') in get_args(DoneActionType) , \
                f'Invalid done action: {getattr(new_cls, 'done_action')}, valid done actions are {get_args(DoneActionType)}'
            assert getattr(new_cls, 'color') in get_args(EncodedColor) , \
                f'Invalid color: {getattr(new_cls, 'color')}, valid colors are {get_args(EncodedColor)}'
            cls.registry[name] = new_cls
        return new_cls

class QuickCallButton(ButtonOperation , metaclass = QuickCallButtonMeta):
    """Abstract base for a single button in the :class:`ControlPanel` action bar.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.

    Encoded colors:
    - purple
    - red
    - green
    - blue
    - orange
    - gray
    - yellow
    - pink
    - gold
    - cyan
    """
    button_title : str = ''
    default_help : str = ''
    done_action : DoneActionType = 'pause'
    color : EncodedColor = 'green'
    research : bool = False

    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        self.update(help = self.default_help)

    @property
    def title(self) -> str:
        if self.button_title:
            return self.button_title
        return self.key.replace('-', ' ').title()

    def run(self) -> None:
        raise NotImplementedError('run method is not implemented in QuickCallButton')

    @abstractmethod
    def script_string(self) -> str:
        """Run the script."""

    @property
    def button_key(self) -> str:
        """Get the key for the button."""
        return f"quick-call-{self.color}-{self.key}-{"disabled" if self.disabled else "enabled"}"

    def call_shell_run(self) -> None:
        """Call the shell run."""
        from src.proj.util.shell import Shell
        script_strings = self.script_string().strip().split('\n')
        script_string = ';'.join([s.strip() for s in script_strings])
        Shell.open(
            ["uv" , "run" , "--frozen" , "python" , "-c" , script_string], 
            done_action=self.done_action,  title = self.title , as_from_workspace='QuickCallButtons'
        )

    def show(self) -> None:
        """Render the button + label into the persistent panel placeholder slot."""
        import streamlit as st
        with st.container():
            st.button(self.icon, key=f'{self.button_key}-button' , help = self.help , disabled = self.disabled , on_click = self.call_shell_run)
            self.render_title(font_size = 11 , uppercase = False)

    @classmethod
    def get_buttons(cls) -> list[QuickCallButton]:
        """Get the buttons dictionary."""
        from importlib import import_module
        import_module('src.api.interactive.util.quick_calls.buttons')
        buttons = [qcb() for qcb in cls.registry.values()]
        return buttons