"""
define common operations for the interactive app to use.
"""
from __future__ import annotations
import streamlit as st
from abc import abstractmethod , ABC
from dataclasses import dataclass , field
from typing import Any , final

from src.proj import BaseClass

@dataclass
class ButtonStatus:
    """The status of the button operation."""
    disabled : bool = False
    help : str = ""
    state : str = ""
    kwargs : dict[str , Any] = field(default_factory = dict)

    def __bool__(self) -> bool:
        return not self.disabled

    def reset(self) -> None:
        self.disabled = False
        self.help = ""
        self.state = ""
        self.kwargs.clear()

    def update(self , disabled : bool | None = None, help : str | None = None , state : str | None = None , **kwargs) -> None:
        if disabled is not None:
            self.disabled = disabled
        if help is not None:
            self.help = help
        if state is not None:
            self.state = state
        self.kwargs.update(kwargs)

class ButtonOperation(ABC , BaseClass.BoundLogger):
    """Abstract base for a single button operation.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """
    key : str = ""
    icon : str = ''
    title : str = ''

    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        self.status = ButtonStatus()

    @property
    def disabled(self) -> bool:
        """The status of the button operation."""
        return self.status.disabled

    @property
    def help(self) -> str:
        """The help of the button operation."""
        return self.status.help

    @property
    def button_key(self) -> str:
        """The button key of the button operation."""
        return f"{self.key}-{"disabled" if self.disabled else "enabled"}-{self.status.state}".replace(' ', '-')

    @abstractmethod
    def run(self) -> None:
        """Run the common operation."""
        ...

    def get(self , key : str) -> Any:
        """Get the value of the key from the status."""
        return self.status.kwargs.get(key)

    def reset(self):
        self.status.reset()
        return self

    @final
    def update(self , disabled : bool | None = None, help : str | None = None , state : str | None = None , **kwargs):
        self.status.update(disabled = disabled, help = help, state = state, **kwargs)
        return self

    def render_title(self , font_size : int = 12 , uppercase : bool = True) -> None:
        """Render the title of the button operation."""
        title = self.title.upper() if uppercase else self.title.title()
        body = f"""
            <div style="margin-bottom: 0px;margin-top: -10px;padding: 0 0 20px 0;font-size: {font_size}px;font-weight: 600;
                white-space: nowrap;">{title}</div>"""       
        st.markdown(body , unsafe_allow_html = True)