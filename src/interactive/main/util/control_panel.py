"""The control panel for the interactive app, will show at the top of every page.

Key objects:

* :class:`ControlPanelButton` / :class:`ControlPanel` — the shared action
  bar rendered at the top of every page via :meth:`SessionControl.get_control_panel`.
"""
from __future__ import annotations
import streamlit as st
import subprocess
import time

from abc import abstractmethod , ABC
from src.proj import Proj , MACHINE , PATH , Const
from src.proj.util import Options
from src.interactive.backend import ScriptRunner
from .session_control import SC

class ControlPanelButton(ABC):
    """Abstract base for a single button in the :class:`ControlPanel` action bar.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """
    key : str = ''
    icon : str = ''
    title : str = ''

    @abstractmethod
    def button(self , script_key : str | None = None) -> None:
        """Render the Streamlit button widget for this action.

        Args:
            script_key: The currently active script key, or ``None`` when on
                an intro page.
        """
        ...

    def refresh(self , *args , **kwargs) -> None:
        """Redraw the button with updated state (override in subclasses as needed)."""
        pass

    def show(self , script_key : str | None = None) -> None:
        """Render the button + label into the persistent panel placeholder slot."""
        if self.key not in st.session_state:
            st.session_state[self.key] = st.empty()
        with st.session_state[self.key]:
            with st.container():
                self.button(script_key = script_key)
                self.print_title()

    def print_title(self) -> None:
        """Render the small capitalised label below the button icon."""
        body = f"""
        <div style="
            margin-bottom: 0px;
            margin-top: -10px;
            padding: 0 0 20px 0;
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
        ">{self.title.upper()}</div>
        """       
        st.markdown(body , unsafe_allow_html = True)

class ScriptRunnerRunButton(ControlPanelButton):
    """Button that submits the current script to the task queue.

    Rendered as disabled (greyed) when no script is selected or required
    parameters are missing; enabled (green) otherwise.
    """
    key = f"script-runner-run"
    icon = f":material/mode_off_on:"
    title = f"Run Script"

    def button(self , script_key : str | None = None):
        help = f"Please Choose a Script to Run First" if script_key is None else f"Please Fill Required Parameters"
        st.button(self.icon, key=f'{self.key}-disabled' , help = help)

    def refresh(self , runner : ScriptRunner):
        with st.session_state[self.key]:
            if SC.param_inputs_form is None:
                raise ValueError("ParamInputsForm is not initiated")
            params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None
            
            if SC.get_script_runner_validity(params):
                disabled = False
                preview_cmd = SC.get_script_runner_cmd(runner , params)
                if preview_cmd: 
                    help_text = preview_cmd
                else:
                    help_text = f"Parameters valid, run {runner.script_key}"
                button_key = f"{self.key}-enabled-{runner.script_key}"
            else:
                disabled = True
                help_text = f"Parameters invalid, please check required ones"
                button_key = f"{self.key}-disabled-{runner.script_key}"

            with st.container():
                st.button(self.icon, key=button_key , 
                        help = help_text , disabled = disabled , 
                        on_click = SC.click_script_runner_run , args = (runner, params)) 
                self.print_title()

class GlobalScriptLatestTaskButton(ControlPanelButton):
    """Button that navigates to the latest task across all scripts."""
    key = f"global-script-latest-task"
    icon = f":material/reply_all:"
    title = f"Latest for All"

    def button(self , script_key : str | None = None):
        item = SC.get_latest_task_item()
        if item is None:
            st.button(self.icon, key=f"{self.key}-disabled" , 
                    help = "Please Run a Task First" , disabled = True)
        else:
            if st.button(self.icon, key=f"{self.key}-enabled-{item.id}" , 
                        help = f":blue[**Show Latest Task**]: {item.id}" , 
                        on_click = SC.click_show_complete_report , args = (item,) ,
                        disabled = False):
                if SC.current_page_name != repr(item.script_key):
                    from src.interactive.main.util.page import get_script_page

                    meta = get_script_page(item.script_key)
                    if meta:
                        st.switch_page(meta['page'])
                else:
                    #from .script_detail import show_report_main
                    #show_report_main(SC.get_script_runner(item.script_key))
                    st.rerun()

class CurrentScriptLatestTaskButton(ControlPanelButton):
    """Button that shows the latest task for the currently displayed script."""
    key = f"current-script-latest-task"
    icon = f":material/reply:"
    title = f"Current Latest"

    def button(self , script_key : str | None = None):
        item = SC.get_latest_task_item(script_key) if script_key is not None else None
        if item is None:
            st.button(self.icon, key=f"{self.key}-disabled" , 
                        help = "Please Run a Task of This Script First" if script_key is not None else "Please Choose a Script First" , disabled = True)
        else:
            if st.button(self.icon, key=f"{self.key}-enabled-{item.id}" , 
                        help = f":blue[**Show Latest Task of This Script**]: {item.id}" , 
                        on_click = SC.click_show_complete_report , args = (item,) ,
                        disabled = False):
                #from .script_detail import show_report_main
                #show_report_main(SC.get_script_runner(item.script_key))
                st.rerun()

class RebootButton(ControlPanelButton):
    """Button that reboots the streamlit app by hot reloading all the modules and clearing the cache."""
    key = f"control-reboot-app"
    icon = f":material/restart_alt:"
    title = f"Reboot App"

    def button(self , script_key : str | None = None):
        st.button(self.icon, key=f"{self.key}-enabled" , help = "Hot Reload the Current Script" , disabled = False, 
                  on_click = self.reboot_app)

    @classmethod
    def reboot_app(cls):
        from src.proj.util.func import streamlit_hot_reload
        streamlit_hot_reload(PATH.main.joinpath('src') , rerun = False)
        time.sleep(1)
        st.rerun()

class ControlRefreshInteractiveButton(ControlPanelButton):
    """Button that regenerates all script-detail pages and reinitialises the session."""
    key = f"control-refresh-interactive"
    icon = f":material/refresh:"
    title = f"Refresh All"

    def button(self , script_key : str | None = None):
        st.button(self.icon, key=f"{self.key}-enabled" , help = "Refresh Task Queue / Options / Scripts" , 
                  on_click = self.refresh_all , disabled = False)
 
    def refresh_all(self):
        with st.spinner("Refreshing..."):
            with Proj.silence:
                Options.update()
        SC.rerun()
        st.rerun()

class ControlGitClearPullButton(ControlPanelButton):
    """Button that resets local changes and pulls the latest code from remote.

    Disabled automatically on coding platforms (``MACHINE.platform_coding``).
    """
    key = f"control-git-clear-pull"
    icon = f":material/cloud_download:"
    title = f"Git Pull"

    def button(self , script_key : str | None = None):
        if MACHINE.platform_coding:
            st.button(self.icon, key=f"{self.key}-disabled" , help = f"Git Pull is not available on coding platform {MACHINE.name}" , disabled = True)
        else:
            st.button(self.icon, key=f"{self.key}-enabled" , help = "Clear Local Changes and Pull Latest Code" , disabled = False, on_click = self.clear_git_pull)
        
    def clear_git_pull(self):
        if MACHINE.platform_coding:
            raise ValueError(f"Git Pull is not available on coding platform {MACHINE.name}")
        else:
            import shutil
            from src.proj import PATH , Logger

            subprocess.run(['git', 'reset', '--hard', 'HEAD'], check=True)
            subprocess.run(['git', 'clean', '-fd'], check=True)
            subprocess.run(['git', 'pull'], check=True)
            
            for folder in [*PATH.main.joinpath('src').rglob('*/') , *PATH.main.joinpath('configs').rglob('*/')][::-1]:
                if folder.is_dir() and not [x for x in folder.iterdir() if x.name != '__pycache__']:
                    subfiles = [x for x in folder.rglob('*') if x.is_file()]
                    if not len(subfiles):
                        Logger.stdout(f"Removing empty folder: {folder}")
                        folder.rmdir()
                    else:
                        if all([x.suffix == '.pyc' for x in subfiles]):
                            Logger.stdout(f"Removing folder with only pyc files: {folder}")
                            shutil.rmtree(folder)
                        else:
                            Logger.error(f"Error removing folder: {folder}:")
                            Logger.error(f"Subfiles: {subfiles}")
            Logger.success("Git Pull Finished")

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
        st.toggle('**:blue[Max Verbosity]**', value=False , key = 'global-settings-max-vb' , 
                help="""Should use max verbosity or min? Not selected will use default.""")
        st.toggle('**:blue[Disable Email]**', value=False , key = 'global-settings-disable-email'  , 
                help="""If email after the script is complete? Not selected will use script header value.""")
        st.toggle("**:blue[Silent Run]**", value=False , key = 'global-settings-silent-run'  , 
                help="""Should the script run silently? Not selected will use script header value.""")

    def show(self , script_key : str | None = None) -> None:
        """Render the button + label into the persistent panel placeholder slot."""
        if self.key not in st.session_state:
            st.session_state[self.key] = st.empty()
        with st.session_state[self.key]:
            with st.popover(f'**{self.title}**' , icon = self.icon , width = 'stretch'):
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
    icon = f":material/info:"
    title = f"**Intro**"

    def popover(self , script_key : str | None = None) -> None:
        """Render icon-button shortcuts for each intro page in the sidebar."""
        from src.interactive.main.util.page import intro_pages
        pages = intro_pages()
        
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
    icon = f":material/settings:"
    title = f"**Settings**"

    def popover(self , script_key : str | None = None) -> None:
        st.toggle('**:blue[Max Verbosity]**', value=False , key = 'global-settings-max-vb' , 
                help="""Should use max verbosity or min? Not selected will use default.""")
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
                    from src.interactive.main.util.page import get_script_page

                    meta = get_script_page(item.script_key)
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

class ControlPanel:
    """Horizontal action bar rendered at the top of every app page.

    Contains a fixed set of :class:`ControlPanelButton` instances plus a
    settings popover for global run toggles (verbosity, email, silent mode).
    """
    control_panel_key = "control-panel"
    buttons : dict[str, ControlPanelButton] = {
        'script-runner-run' : ScriptRunnerRunButton(),
        'control-reboot-app' : RebootButton(),
        'control-refresh-interactive' : ControlRefreshInteractiveButton(),
        'control-git-clear-pull' : ControlGitClearPullButton(),
    }
    popovers : dict[str, ControlPanelPopover] = {
        'intro-page' : IntroPagePopover(),
        'global-settings' : GlobalSettingsPopover(),
        'more-buttons' : MoreButtonsPopover(),
    }

    @property
    def area_columns(self) -> tuple[float, float, float]:
        width_ratio = Const.Pref.interactive.get('control_panel_width_ratio' , 0.8)
        return (1 - width_ratio) / 2 , width_ratio , (1 - width_ratio) / 2
    
    def show(self , script_key : str | None = None) -> None:
        """Render the full control panel (buttons + settings popover).

        Args:
            script_key: Passed through to each button so they can
                enable/disable themselves based on whether a script is active.
        """
        with st.container(key = f"{self.control_panel_key}-container"):
            self.show_buttons(script_key = script_key)
            self.show_popovers(script_key = script_key)

    def show_buttons(self , script_key : str | None = None) -> None:
        """Render the control panel buttons area.

        Args:
            script_key: Passed through to each button so they can
                enable/disable themselves based on whether a script is active.
        """
        _ , area , _ = st.columns(self.area_columns , gap = 'small' , vertical_alignment = 'center')
        min_cols , max_cols = Const.Pref.interactive.get('control_panel_buttons_columns' , [3 , 5])
        with area.container(key = f"{self.control_panel_key}-buttons"):
            nrows = (len(self.buttons) / max_cols).__ceil__()
            ncols = max(min(len(self.buttons) , max_cols) , min_cols)
            buttons = list(self.buttons.values())
            for irow in range(nrows):
                cols = st.columns(ncols , gap = 'small' , vertical_alignment = 'center')
                for icol , col in zip(range(ncols), cols):
                    if irow * ncols + icol >= len(buttons):
                        break
                    button = buttons[irow * ncols + icol]
                    with col:
                        button.show(script_key = script_key)

    def show_popovers(self , script_key : str | None = None) -> None:
        """Render the control panel popovers area.
        """
        _ , area , _ = st.columns(self.area_columns , gap = 'small' , vertical_alignment = 'center')
        min_cols , max_cols = Const.Pref.interactive.get('control_panel_popovers_columns' , [2 , 3])

        with area.container(key = f"{self.control_panel_key}-popovers"):
            nrows = (len(self.popovers) / max_cols).__ceil__()
            ncols = max(min(len(self.popovers) , max_cols) , min_cols)
            popovers = list(self.popovers.values())
            for irow in range(nrows):
                cols = st.columns(ncols , gap = 'large' , vertical_alignment = 'center')
                for icol , col in zip(range(ncols), cols):
                    if irow * ncols + icol >= len(popovers):
                        break
                    popover = popovers[irow * ncols + icol]
                    with col:
                        popover.show(script_key = script_key)