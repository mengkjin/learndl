"""Buttons for the control panel. Css style defined in templates/css/interactive/button_groups.template"""
from __future__ import annotations
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from abc import abstractmethod
from typing import TYPE_CHECKING
from src.proj import Proj , MACHINE , Options
from src.api.interactive.util.session_control import SC
from src.api.interactive.util.components.operations import ButtonOperation

if TYPE_CHECKING:
    from src.api.util.backend import TaskItem , ScriptRunner

__all__ = ['ControlPanelButton']

class ControlPanelButton(ButtonOperation):
    """Abstract base for a single button in the :class:`ControlPanel` action bar.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """

    @property
    def runner(self) -> ScriptRunner | None:
        return self.get('runner')

    @property
    def script_key(self) -> str | None:
        return self.get('script_key')

    @abstractmethod
    def update_status(self):
        """Update the status of the button."""
        pass

    def cold_start(self , script_key : str | None = None):
        """Cold start the button when the page is loaded."""
        self.reset().update(script_key = script_key)
        self.update_status()
        self.show()

    def refresh(self , **kwargs) -> None:
        """Redraw the button with updated state (override in subclasses as needed)."""
        pass

    def render_area(self) -> DeltaGenerator:
        """Render the area for the button."""
        area_key = f"cpb-{self.key}-area"
        if area_key not in st.session_state:
            st.session_state[area_key] = st.empty()
        return st.session_state[area_key]

    def render_button(self) -> None:
        """Render the button."""
        st.button(self.icon, key=self.button_key , help = self.help , disabled = self.disabled , on_click = self.run)
        self.render_title(font_size = 12 , uppercase = True)

    def show(self) -> None:
        """Render the button + label into the persistent panel placeholder slot."""
        with self.render_area().container():
            self.render_button()

class ScriptRunnerRunButton(ControlPanelButton):
    """Button that submits the current script to the task queue."""
    key = f"script-runner-run"
    icon = f":material/mode_off_on:"
    title = f"Run Script"

    def update_status(self):
        """Update the status of the button."""
        if self.script_key is None:
            return self.update(True , "Please Choose a Script to Run First" , "no-script-key")
        if self.runner is None:
            return self.update(True , "Please Supply a Script Runner First" , "no-runner")
        if self.runner.blacklisted:
            return self.update(True , "Script is blacklisted on this machine" , "blacklisted")
        if SC.param_inputs_form is None:
            return self.update(True , "Please Fill Required Parameters" , "no-param-form")
        params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None
        if SC.get_script_runner_validity(params):
            preview_cmd = SC.get_script_runner_cmd(self.runner , params)
            help = preview_cmd if preview_cmd else f"Parameters valid, run {self.script_key}"
            self.update(False , help , self.script_key , params = params)
        else:
            help = "Parameters invalid, please check required ones"
            self.update(True , help , self.script_key , params = params)
        return self

    def refresh(self , runner : ScriptRunner , **kwargs) -> None:
        raw_state = self.status.state
        self.update(runner = runner).update_status()
        if self.status.state != raw_state:
            self.show()

    def run(self) -> None:
        SC.click_script_runner_run(self.runner , self.get('params'))

class GlobalScriptLatestTaskButton(ControlPanelButton):
    """Button that navigates to the latest task across all scripts."""
    key = f"global-script-latest-task"
    icon = f":material/reply_all:"
    title = f"Global Last Task"

    def update_status(self):
        item = SC.get_latest_task_item()
        if item is None:
            self.update(True , "Please Run a Task First" , "no-item")
        else:
            self.update(False , f":blue[**Show Latest Task**]: {item.id}" , item.id , item = item)

    def run(self) -> None:
        item : TaskItem | None = self.get('item')
        if item is None:
            return
        SC.click_show_complete_report(item)
        if SC.current_page_name != repr(item.script_key):
            meta = SC.get_page(item.script_key)
            if meta:
                st.switch_page(meta['page'])
        else:
            st.rerun()

class CurrentScriptLatestTaskButton(ControlPanelButton):
    """Button that shows the latest task for the currently displayed script."""
    key = f"current-script-latest-task"
    icon = f":material/reply:"
    title = f"Current Last Task"
    
    def update_status(self):
        """Update the status of the button."""
        if self.script_key is None:
            return self.update(True , "Please Choose a Script First" , "no-script-key")
        item = SC.get_latest_task_item(self.script_key)
        if item is None:
            self.update(True , "Please Run a Task of This Script First" , "no-item")
        else:
            self.update(False , f":blue[**Show Latest Task of This Script**]: {item.id}" , item.id , item = item)
    
    def run(self) -> None:
        item : TaskItem | None = self.get('item')
        if item is None:
            return
        SC.click_show_complete_report(item)
        st.rerun()

class ControlRefreshInteractiveButton(ControlPanelButton):
    """Button that regenerates all script-detail pages and reinitialises the session."""
    key = f"refresh-interactive"
    icon = f":material/directory_sync:"
    title = f"Refresh All"

    def update_status(self):
        """Update the status of the button."""
        self.update(False , "Refresh Task Queue / Options / Scripts" , "ok")

    def run(self) -> None:
        with st.spinner("Refreshing...") , Proj.silence:
            Options.update()
        SC.bump_backend_refresh_epoch()
        SC.rerun()
        st.rerun()

class ControlGitClearPullButton(ControlPanelButton):
    """Button that resets local changes and pulls the latest code from remote.

    Disabled automatically on coding platforms (``MACHINE.platform_coding``).
    """
    key = f"git-clear-pull"
    icon = f":material/cloud:"
    title = f"Git Pull"

    def update_status(self):
        if MACHINE.platform_coding:
            self.update(True, f"Git Pull is not available on coding platform {MACHINE.name}" , "platform-coding")
        else:
            self.update(False, "Reset Local Changes and Pull Latest Code" , "ok")

    def run(self) -> None:
        if MACHINE.platform_coding:
            raise ValueError(f"Git Pull is not available on coding platform {MACHINE.name}")
        SC.bump_backend_refresh_epoch()
        from src.api.calls.source_code import GitClearPull
        GitClearPull.go(verbose_level=1)
        st.rerun()

class ControlGitClearPullRunButton(ControlPanelButton):
    """Button that resets local changes and pulls the latest code from remote.

    Disabled automatically on coding platforms (``MACHINE.platform_coding``).
    """
    key = f"pull-and-run"
    icon = f":material/cloud_sync:"
    title = f"Pull & Run"

    def update_status(self):
        """Update the status of the button."""
        if MACHINE.platform_coding:
            return self.update(True, f"Git Pull is not available on coding platform {MACHINE.name}" , "platform-coding")
        if self.script_key is None:
            return self.update(True , "Please Choose a Script to Run First" , "no-script-key")
        if self.runner is None:
            return self.update(True , "Please Supply a Script Runner First" , "no-runner")
        if SC.param_inputs_form is None:
            return self.update(True , "Please Fill Required Parameters" , "no-param-form")
        params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None
        if SC.get_script_runner_validity(params):
            preview_cmd = SC.get_script_runner_cmd(self.runner , params)
            help = preview_cmd if preview_cmd else f"Parameters valid, run {self.script_key}"
            self.update(False , help , self.script_key , params = params)
        else:
            help = "Parameters invalid, please check required ones"
            self.update(True , help , self.script_key , params = params)
        return self

    def refresh(self , runner : ScriptRunner , **kwargs) -> None:
        raw_state = self.status.state
        self.update(runner = runner).update_status()
        if self.status.state != raw_state:
            self.show()

    def run(self) -> None:
        if MACHINE.platform_coding:
            raise ValueError(f"Git Pull is not available on coding platform {MACHINE.name}")
        script_key = self.script_key
        runner = self.runner
        if script_key is None or runner is None:
            return
        st.session_state['pending_pull_and_run'] = {
            'script_key': script_key,
            'params': self.get('params'),
        }
        SC.bump_backend_refresh_epoch()
        try:
            from src.api.calls.source_code import GitClearPull
            GitClearPull.go(verbose_level=1)
        except Exception:
            st.session_state.pop('pending_pull_and_run', None)
            raise
        st.rerun()