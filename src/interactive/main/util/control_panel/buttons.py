"""Buttons for the control panel. Css style defined in templates/css/interactive/button_groups.template"""
from __future__ import annotations
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from abc import abstractmethod
from typing import TYPE_CHECKING , Literal

from src.interactive.main.util.components import common_operations as CO

if TYPE_CHECKING:
    from src.interactive.backend import TaskItem , ScriptRunner

def _print_title(title : str) -> None:
    """Render the small capitalised label below the button icon."""
    body = f"""
    <div style="
        margin-bottom: 0px;
        margin-top: -10px;
        padding: 0 0 20px 0;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    ">{title.upper()}</div>
    """       
    st.markdown(body , unsafe_allow_html = True)

class ControlPanelButton(CO.CommonOperation):
    """Abstract base for a single button in the :class:`ControlPanel` action bar.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """
    key : str = ''
    icon : str = ''
    title : str = ''

    @abstractmethod
    def button_key(self , status : CO.OperationStatus , **kwargs) -> str:
        """Get the key for the button."""
        return f"{self.key}-{"disabled" if status.disabled else "enabled"}"

    def refresh(self , *args , **kwargs) -> None:
        """Redraw the button with updated state (override in subclasses as needed)."""
        pass

    def render_area(self) -> DeltaGenerator:
        """Render the area for the button."""
        area_key = f"cpb-{self.key}-area"
        if area_key not in st.session_state:
            st.session_state[area_key] = st.empty()
        return st.session_state[area_key]

    def render_button(self , **kwargs) -> None:
        """Render the button."""
        status = self.status
        button_key = self.button_key(status , **kwargs)
        st.button(self.icon, key=button_key , help = status.help , disabled = status.disabled , on_click = self.run , args = (status,))

    def show(self , script_key : str | None = None) -> None:
        """Render the button + label into the persistent panel placeholder slot."""
        with self.render_area().container():
            self.render_button(script_key = script_key)
            _print_title(self.title)

class ScriptRunnerRunButton(ControlPanelButton , CO.RunCurrentScript):
    """Button that submits the current script to the task queue."""
    key = f"script-runner-run"
    icon = f":material/mode_off_on:"
    title = f"Run Script"

    def button_key(self , status : CO.OperationStatus , stage : Literal[0,1] = 0 , **kwargs) -> str:
        if stage == 0:
            return f"{self.key}-{"disabled" if status.disabled else "enabled"}-not-refreshed"
        elif stage == 1:
            runner : ScriptRunner = status.kwargs['runner']
            return f"{self.key}-{"disabled" if status.disabled else "enabled"}-{runner.script_key}"
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def refresh(self , runner : ScriptRunner):
        self.update_kwargs(runner = runner)
        with self.render_area().container():
            self.render_button(stage = 1)

class GlobalScriptLatestTaskButton(ControlPanelButton , CO.GlobalScriptLatestTask):
    """Button that navigates to the latest task across all scripts."""
    key = f"global-script-latest-task"
    icon = f":material/reply_all:"
    title = f"Global Last Task"

    def button_key(self , status : CO.OperationStatus , **kwargs) -> str:
        item : TaskItem | None = status.kwargs.get('item')
        return f"{self.key}-{"disabled" if status.disabled else "enabled"}-{"no-id" if item is None else item.id}"

class CurrentScriptLatestTaskButton(ControlPanelButton , CO.CurrentScriptLatestTask):
    """Button that shows the latest task for the currently displayed script."""
    key = f"current-script-latest-task"
    icon = f":material/reply:"
    title = f"Current Last Task"

    def button_key(self , status : CO.OperationStatus , **kwargs) -> str:
        item : TaskItem | None = status.kwargs.get('item')
        return f"{self.key}-{"disabled" if status.disabled else "enabled"}-{"no-id" if item is None else item.id}"

class RebootButton(ControlPanelButton , CO.RebootApp):
    """Button that reboots the streamlit app by hot reloading all the modules and clearing the cache."""
    key = f"control-reboot-app"
    icon = f":material/restart_alt:"
    title = f"Reboot App"

    def button_key(self , status : CO.OperationStatus , **kwargs) -> str:
        return f"{self.key}-{"disabled" if status.disabled else "enabled"}"

class ControlRefreshInteractiveButton(ControlPanelButton , CO.RefreshAll):
    """Button that regenerates all script-detail pages and reinitialises the session."""
    key = f"control-refresh-interactive"
    icon = f":material/directory_sync:"
    title = f"Refresh All"

    def button_key(self , status : CO.OperationStatus , **kwargs) -> str:
        return f"{self.key}-{"disabled" if status.disabled else "enabled"}"

class ControlGitClearPullButton(ControlPanelButton , CO.GitClearPull):
    """Button that resets local changes and pulls the latest code from remote.

    Disabled automatically on coding platforms (``MACHINE.platform_coding``).
    """
    key = f"control-pull-and-run"
    icon = f":material/cloud:"
    title = f"Git Pull"

    def button_key(self , status : CO.OperationStatus , **kwargs) -> str:
        return f"{self.key}-{"disabled" if status.disabled else "enabled"}"

class ControlGitClearPullRunButton(ControlPanelButton , CO.GitClearPullRun):
    """Button that resets local changes and pulls the latest code from remote.

    Disabled automatically on coding platforms (``MACHINE.platform_coding``).
    """
    key = f"control-git-clear-pull"
    icon = f":material/cloud_sync:"
    title = f"Pull & Run"

    def button_key(self , status : CO.OperationStatus , stage : Literal[0,1] = 0 , **kwargs) -> str:
        if stage == 0:
            return f"{self.key}-{"disabled" if status.disabled else "enabled"}-not-refreshed"
        elif stage == 1:
            runner : ScriptRunner = status.kwargs['runner']
            return f"{self.key}-{"disabled" if status.disabled else "enabled"}-{runner.script_key}"
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def refresh(self , runner : ScriptRunner):
        self.update_kwargs(runner = runner)
        with self.render_area().container():
            self.render_button(stage = 1)