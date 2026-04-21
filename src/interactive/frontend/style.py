"""
Custom CSS injection for the interactive Streamlit application.

Loads CSS template strings from the project's template directory and applies
them to the page via :func:`style`.
"""
import streamlit as st
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
        self.css_list : list[str] = [getattr(self , css)() for css in ['basic' , 'special_expander' , 'classic_remove' , 'multi_select' , 'custom']]
        

    def apply(self) -> None:
        """Inject all CSS strings into the Streamlit page via ``st.markdown``."""
        css_str = '\n'.join(self.css_list)
        st.markdown(f'''
        <style>
        {css_str}
        </style>
        ''' , unsafe_allow_html = True)

    def basic(self) -> str:
        """Return the rendered ``basic`` CSS template string."""
        return self.templates['basic'].substitute()

    def special_expander(self) -> str:
        """Return the rendered ``special_expander`` CSS template string."""
        return self.templates['special_expander'].substitute()

    def classic_remove(self) -> str:
        """Return the rendered ``classic_remove`` CSS template string."""
        return self.templates['classic_remove'].substitute()

    def custom(self) -> str:
        """Return the rendered ``custom`` CSS template string."""
        return self.templates['custom'].substitute()

    def multi_select(self , label_size : int = 16 , item_size : int = 16 , popover_size : int = 14) -> str:
        """Return the rendered ``multi_select`` CSS template with configurable font sizes.

        Parameters
        ----------
        label_size:
            Font size (px) for the multi-select label.
        item_size:
            Font size (px) for individual items in the dropdown.
        popover_size:
            Font size (px) for the popover container.
        """
        return self.templates['multi_select'].substitute(label_size = label_size , item_size = item_size , popover_size = popover_size)


def style() -> None:
    """Instantiate :class:`CustomCSS` and apply all templates to the current page."""
    css = CustomCSS()
    css.apply()
