"""
Streamlit frontend utilities for the interactive application.

ColoredText
    String subclass that auto-applies Streamlit badge colours based on log level.
"""
from __future__ import annotations

class ColoredText(str):
    """String subclass that wraps text in a Streamlit colour badge based on log-level prefix.

    Supports: ``error`` → red, ``warning`` → orange, ``info`` → green,
    ``debug`` → gray, ``critical`` → violet.  Unrecognised messages are returned unchanged.
    """
    def __init__(self , text : str) -> None:
        """Initialise and auto-detect the colour for *text*."""
        self.text = text
        self.color = self.auto_color(self.text)

    def __str__(self) -> str:
        """Return the text wrapped in a Streamlit colour badge, or plain text if no colour applies."""
        if self.color is None:
            return self.text
        else:
            if self.color in ['violet' , 'red']:
                return f":{self.color}[**{self.text}**]"
            else:
                return f":{self.color}[{self.text}]"

    @staticmethod
    def auto_color(message : str) -> str | None:
        """Return the Streamlit colour name for *message*'s log-level prefix, or None."""
        if message.lower().startswith('error'):
            return 'red'
        elif message.lower().startswith('warning'):
            return 'orange'
        elif message.lower().startswith('info'):
            return 'green'
        elif message.lower().startswith('debug'):
            return 'gray'
        elif message.lower().startswith('critical'):
            return 'violet'
        else:
            return None