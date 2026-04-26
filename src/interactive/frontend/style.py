"""
Custom CSS injection for the interactive Streamlit application.

Loads CSS template strings from the project's template directory and applies
them to the page via :func:`style`.
"""
import streamlit as st
from collections import defaultdict
from typing import Any

from src.proj import PATH


class CustomCSS:
    """Loads and applies a set of CSS templates to the active Streamlit page.

    Templates are resolved from the project's ``css/interactive`` template
    directory via :func:`src.proj.PATH.load_templates`.

    Attributes
    ----------
    Templates:
        Class-level dict mapping template name → :class:`~string.Template`.
    """
    

    def __init__(self) -> None:
        """Build the ordered list of CSS strings from all registered templates."""
        self.templates = PATH.load_templates('css' , 'interactive')

    def substitute_kwargs(self , **kwargs) -> dict[str, dict[str, Any]]:
        out = defaultdict(dict)
        out['multi_select'] = {
            'label_size': kwargs.get('label_size', 16),
            'item_size': kwargs.get('item_size', 16),
            'popover_size': kwargs.get('popover_size', 14),
        }
        return out

    def apply(self , **kwargs) -> None:
        """Inject all CSS strings into the Streamlit page via ``st.markdown``."""
        sub_kwargs = self.substitute_kwargs(**kwargs)
        css_str = '\n'.join([template.substitute(sub_kwargs[name]) for name , template in self.templates.items()])
        
        st.markdown(f'''
        <style>
        {css_str}
        </style>
        ''' , unsafe_allow_html = True)

def style() -> None:
    """Instantiate :class:`CustomCSS` and apply all templates to the current page."""
    css = CustomCSS()
    css.apply()
