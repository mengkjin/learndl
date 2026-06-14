"""Popovers for the control panel. Color css defined in templates/css/interactive/popover_colors.template"""
from __future__ import annotations
import streamlit as st

from abc import abstractmethod , ABC

from src.api.interactive.util.session_control import SC


__all__ = ['ControlPanelPopover']

class ControlPanelPopover(ABC):
    """Abstract base for a single button in the :class:`ControlPanel` action bar.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """
    key : str = ''
    icon : str = ''
    title : str = ''

    @abstractmethod
    def popover(self , script_key : str | None = None) -> None:
        """Render the Streamlit popover widget for this action.

        Args:
            script_key: The currently active script key, or ``None`` when on
                an intro page.
        """
        st.toggle('**:blue[Some toggle]**', value=False , key = 'global-settings-some-toggle' , 
                help="""Some help text.""")

    def show(self , script_key : str | None = None) -> None:
        """Render the button + label into the persistent panel placeholder slot."""
        if self.key not in st.session_state:
            st.session_state[self.key] = st.empty()
        with st.session_state[self.key].container(horizontal_alignment = 'center'):
            with st.popover(f'**{self.title}**' , icon = self.icon , width = 'content'):
                with st.container(key = f"control-panel-popover-container-{self.key}"):
                    self.popover(script_key = script_key)

class IntroPagePopover(ControlPanelPopover):
    """
    Popover that shows the intro page links.
    This popover is shown on every page.

    Args:
        script_key: The currently active script key, or ``None`` when on
            an intro page.
    """
    key = f"intro-page-popover"
    icon = f":material/link:"
    title = f"**Intro**"

    def popover(self , script_key : str | None = None) -> None:
        """Render icon-button shortcuts for each intro page in the sidebar."""
        pages = SC.intro_pages
        
        for name , page in pages.items():
            if st.button(f'**{page["label"]}**' , icon = page["icon"] , key = f"control-panel-{self.key}-{name}" ,
                         help = f""":blue[**{page['label'].title()}**] - {page['help']}""" , width = 'stretch' , type = 'tertiary'):
                st.switch_page(page['page'])
            
class GlobalSettingsPopover(ControlPanelPopover):
    """
    Popover that shows the global settings.
    This popover is shown on every page.

    Args:
        script_key: The currently active script key, or ``None`` when on
            an intro page.
    """
    key = f"global-settings-popover"
    icon = f":material/settings_input_component:"
    title = f"**Setting**"

    def popover(self , script_key : str | None = None) -> None:
        st.toggle('**:blue[Debug Mode]**', value=False , key = 'global-settings-debug-mode' , 
                help="""Should use debug mode? Not selected will use default False.""")
        st.toggle('**:blue[Max Verbosity]**', value=False , key = 'global-settings-max-vb' , 
                help="""Should use max verbosity or min? Not selected will use default False.""")
        st.toggle('**:blue[Disable Email]**', value=False , key = 'global-settings-disable-email'  , 
                help="""If email after the script is complete? Not selected will use script header value.""")
        st.toggle("**:blue[Silent Run]**", value=False , key = 'global-settings-silent-run'  , 
                help="""Should the script run silently? Not selected will use script header value.""")

class MoreButtonsPopover(ControlPanelPopover):
    """
    Popover that shows the more buttons.
    This popover is shown on every page.

    Args:
        script_key: The currently active script key, or ``None`` when on
            an intro page.
    """
    key = f"more-buttons-popover"
    icon = f":material/more_horiz:"
    title = f"**More**"

    def popover(self , script_key : str | None = None) -> None:
        self.global_script_latest_task_button(script_key = script_key)
        self.current_script_latest_task_button(script_key = script_key)

    def global_script_latest_task_button(self , script_key : str | None = None):
        item = SC.get_latest_task_item()
        icon = f":material/reply_all:"
        title = f"**Latest for All**"
        if item is None:
            st.button(title , icon = icon, disabled = True, help = "Please Run a Task First" , type = 'tertiary')
        else:
            if st.button(title , icon = icon, width = 'stretch',
                         help = f":blue[**Show Latest Task**]: {item.id}" , 
                         on_click = SC.click_show_complete_report , args = (item,), type = 'tertiary'):
                if SC.current_page_name != repr(item.script_key):
                    meta = SC.get_page(item.script_key)
                    if meta:
                        st.switch_page(meta['page'])
                else:
                    st.rerun()

    def current_script_latest_task_button(self , script_key : str | None = None):
        item = SC.get_latest_task_item(script_key) if script_key is not None else None
        icon = ':material/reply:'
        title = f'**Current Latest**'
        if item is None:
            st.button(title , icon = icon , disabled = True, width = 'stretch', type = 'tertiary' ,
                      help = "Please Run a Task of This Script First" if script_key is not None else "Please Choose a Script First")
        else:
            if st.button(title , icon = icon , width = 'stretch', type = 'tertiary' ,
                         help = f":blue[**Show Latest Task of This Script**]: {item.id}" , 
                         on_click = SC.click_show_complete_report , args = (item,)):
                #from .script_detail import show_report_main
                #show_report_main(SC.get_script_runner(item.script_key))
                st.rerun()

class SystemInfoPopover(ControlPanelPopover):
    """
    Popover that shows the system information.
    This popover is shown on every page.

    Args:
        script_key: The currently active script key, or ``None`` when on
            an intro page.
    """
    key = f"system-info-popover"
    icon = f":material/device_thermostat:"
    title = f"**System**"

    def popover(self , script_key : str | None = None) -> None:
        from src.api.interactive.util.components.intro import get_system_info
        options = get_system_info()
        for label , value in options.items():
            cols = st.columns([2,3])
            cols[0].markdown(f"***{label}***")
            cols[1].markdown(f":blue-badge[*{value}*]")