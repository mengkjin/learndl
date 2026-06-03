"""
Quick call buttons for the interactive app, can be directly used in any page.
"""
from __future__ import annotations
import streamlit as st

from abc import abstractmethod , ABCMeta
from typing import Literal , Any

from src.interactive.main.util.components import common_operations as CO

def _print_title(title : str) -> None:
    """Print the title."""
    body = f"""
    <div style="
        margin-bottom: 0px;
        margin-top: -10px;
        padding: 0 0 20px 0;
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
    ">{title.title()}</div>
    """       
    st.markdown(body , unsafe_allow_html = True)

class QuickCallButtonMeta(ABCMeta):
    """Meta class for QuickCallButton."""
    registry : dict[str, type[QuickCallButton] | Any] = {}
    def __new__(cls , name , bases , dct):
        new_cls = super().__new__(cls , name , bases , dct)
        if name != 'QuickCallButton':
            cls.registry[name] = new_cls
        return new_cls

class QuickCallButton(CO.CommonOperation , metaclass = QuickCallButtonMeta):
    """Abstract base for a single button in the :class:`ControlPanel` action bar.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """
    key : str = ''
    icon : str = ''
    help : str = ''

    def __init__(
        self , color : Literal[
            'red' , 'green' , 'blue' , 'orange' , 'purple' , 'gray' , 'yellow' , 'pink' , 'gold' , 'cyan'] | Any = 'green' , 
        **kwargs):
        super().__init__(**kwargs)
        self.color = color

    @property
    def title(self) -> str:
        return self.key.replace('-', ' ').title()

    @property
    def status(self) -> CO.OperationStatus:
        """Get the status of the button."""
        return CO.OperationStatus(False , self.help)

    def run(self , status : CO.OperationStatus) -> None:
        raise NotImplementedError('run method is not implemented in QuickCallButton')

    @abstractmethod
    def script_string(self) -> str:
        """Run the script."""

    def button_key(self , status : CO.OperationStatus , **kwargs) -> str:
        """Get the key for the button."""
        return f"quick-call-{self.color}-{self.key}-{"disabled" if status.disabled else "enabled"}"

    def call_shell_run(self) -> None:
        """Call the shell run."""
        from src.proj.util.shell_opener import Shell
        script_strings = self.script_string().strip().split('\n')
        script_string = ';'.join([s.strip() for s in script_strings])
        Shell.open(
            ["uv" , "run" , "python" , "-c" , script_string], title = self.title , 
            as_from_workspace='QuickCallButtons')

    def show(self , **kwargs) -> None:
        """Render the button + label into the persistent panel placeholder slot."""
        status = self.status
        button_key = self.button_key(status , **kwargs)
        with st.container():
            st.button(self.icon, key=f'{button_key}-button' , help = status.help , disabled = status.disabled , on_click = self.call_shell_run)
            _print_title(self.title)

    @classmethod
    def get_buttons(cls) -> list[QuickCallButton]:
        """Get the buttons dictionary."""
        from importlib import import_module
        import_module('src.interactive.main.util.quick_calls.buttons')
        buttons = [
            cls.registry['TestLogger'](color = 'cyan'),
            cls.registry['CheckConfigFiles'](color = 'cyan'),
            cls.registry['Tensorboard'](color = 'gold'),
            cls.registry['OptunaDashboard'](color = 'gold'),
            cls.registry['Reboot'](color = 'purple'),
            cls.registry['ArchiveModel'](color = 'pink'),
            cls.registry['ResumeModel'](color = 'green'),
            cls.registry['ClearCatcherLogs'](color = 'red'),

        ]
        return buttons