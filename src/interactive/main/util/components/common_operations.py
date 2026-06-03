"""
define common operations for the interactive app to use.
"""
from __future__ import annotations
import streamlit as st
import subprocess
import time

from abc import abstractmethod , ABC
from dataclasses import dataclass , field
from typing import Any , TYPE_CHECKING

from src.proj import Proj , MACHINE , PATH , BaseClass
from src.proj.util.options import Options
from src.interactive.main.util.session_control import SC

if TYPE_CHECKING:
    from src.interactive.backend import TaskItem , ScriptRunner


def _clear_git_pull() -> None:
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
@dataclass
class OperationStatus:
    """The status of the common operation."""
    disabled : bool = True
    help : str = ""
    kwargs : dict[str , Any] = field(default_factory = dict)

    def __bool__(self) -> bool:
        return not self.disabled

    def reset(self) -> None:
        self.disabled = True
        self.help = ""

    def update(self , disabled : bool = True, help : str = "") -> None:
        self.disabled = disabled
        self.help = help

class CommonOperation(ABC , BaseClass.BoundLogger):
    """Abstract base for a single common operation.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """
    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    @property
    def status(self) -> OperationStatus:
        """The title for the common operation."""
        return OperationStatus()

    @classmethod
    @abstractmethod
    def run(cls , status : OperationStatus) -> None:
        """Run the common operation."""
        ...

    def update_kwargs(self , **kwargs):
        self.kwargs = kwargs

    @property
    def runner(self) -> ScriptRunner | None:
        return self.kwargs.get('runner')

class RunCurrentScript(CommonOperation):
    """Submits the current script to the task queue."""
    @property
    def status(self) -> OperationStatus:
        """The key for the common operation."""
        if self.runner is None:
            return OperationStatus(disabled = True, help = "Please Choose a Script to Run First")
        if SC.param_inputs_form is None:
            return OperationStatus(disabled = True, help = "Please Fill Required Parameters")
        params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None

        if SC.get_script_runner_validity(params):
            disabled = False
            preview_cmd = SC.get_script_runner_cmd(self.runner , params)
            if preview_cmd: 
                help_text = preview_cmd
            else:
                help_text = f"Parameters valid, run {self.runner.script_key}"
        else:
            disabled = True
            help_text = f"Parameters invalid, please check required ones"
        return OperationStatus(disabled , help_text , {'params': params , 'runner': self.runner})

    @classmethod
    def run(cls , status : OperationStatus) -> None:
        runner : ScriptRunner = status.kwargs['runner']
        params : dict[str, Any] | None = status.kwargs.get('params')
        SC.click_script_runner_run(runner , params)

class GlobalScriptLatestTask(CommonOperation):
    """Navigates to the latest task across all scripts."""
    @property
    def status(self) -> OperationStatus:
        """The key for the common operation."""
        item = SC.get_latest_task_item()
        if item is None:
            return OperationStatus(True , "Please Run a Task First" , {})
        else:
            return OperationStatus(False , f":blue[**Show Latest Task**]: {item.id}" , {'item': item})

    @classmethod
    def run(cls , status : OperationStatus) -> None:
        item : TaskItem | None = status.kwargs.get('item')
        if item is None:
            return
        SC.click_show_complete_report(item)
        if SC.current_page_name != repr(item.script_key):
            meta = SC.get_page(item.script_key)
            if meta:
                st.switch_page(meta['page'])
        else:
            st.rerun()

class CurrentScriptLatestTask(CommonOperation):
    """Shows the latest task for the currently displayed script."""
    @property
    def status(self) -> OperationStatus:
        """The key for the common operation."""
        item = SC.get_latest_task_item(self.runner.script_key) if self.runner else None
        if self.runner is None:
            return OperationStatus(True , "Please Choose a Script First" , {})
        elif item is None:
            return OperationStatus(True , "Please Run a Task of This Script First" , {})
        else:
            return OperationStatus(False , f":blue[**Show Latest Task of This Script**]: {item.id}" , {'item': item , 'runner': self.runner})

    @classmethod
    def run(cls , status : OperationStatus) -> None:
        item : TaskItem | None = status.kwargs.get('item')
        if item is None:
            return
        SC.click_show_complete_report(item)
        st.rerun()

class RebootApp(CommonOperation):
    """Reboots the streamlit app by hot reloading all the modules and clearing the cache."""
    @property
    def status(self) -> OperationStatus:
        """The key for the common operation."""
        return OperationStatus(False, "Hot Reload the Current Script")

    @classmethod
    def run(cls , status : OperationStatus) -> None:
        from src.proj.util.deepreload import streamlit_hot_reload
        streamlit_hot_reload(PATH.main.joinpath('src') , rerun = False)
        time.sleep(1)
        st.rerun()

class RefreshAll(CommonOperation):
    """Regenerates all script-detail pages / options / scripts and reinitialises the session."""
    @property
    def status(self) -> OperationStatus:
        """The key for the common operation."""
        return OperationStatus(False, "Refresh Task Queue / Options / Scripts")

    @classmethod
    def run(cls , status : OperationStatus) -> None:
        with st.spinner("Refreshing...") , Proj.silence:
            Options.update()
        SC.rerun()
        st.rerun()

class GitClearPull(CommonOperation):
    """Resets local changes and pulls the latest code from remote."""
    @property
    def status(self) -> OperationStatus:
        """The key for the common operation."""
        if MACHINE.platform_coding:
            return OperationStatus(True, f"Git Pull is not available on coding platform {MACHINE.name}")
        else:
            return OperationStatus(False, "Reset Local Changes and Pull Latest Code")

    @classmethod
    def run(cls , status : OperationStatus) -> None:
        if MACHINE.platform_coding:
            raise ValueError(f"Git Pull is not available on coding platform {MACHINE.name}")
        else:
            _clear_git_pull()

class GitClearPullRun(CommonOperation):
    """Runs the git clear pull operation and run the current script."""
    @property
    def status(self) -> OperationStatus:
        """The key for the common operation."""
        if MACHINE.platform_coding:
            return OperationStatus(True, f"Git Pull is not available on coding platform {MACHINE.name}")
        if self.runner is None:
            return OperationStatus(True, "Please Choose a Script to Run First")
        if SC.param_inputs_form is None:
            return OperationStatus(True, "Please Fill Required Parameters")
        params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None

        if SC.get_script_runner_validity(params):
            disabled = False
            preview_cmd = SC.get_script_runner_cmd(self.runner , params)
            if preview_cmd: 
                help_text = preview_cmd
            else:
                help_text = f"Parameters valid, run {self.runner.script_key}"
        else:
            disabled = True
            help_text = f"Parameters invalid, please check required ones"
        return OperationStatus(disabled , help_text , {'params': params , 'runner': self.runner})

    @classmethod
    def run(cls , status : OperationStatus) -> None:
        if MACHINE.platform_coding:
            raise ValueError(f"Git Pull is not available on coding platform {MACHINE.name}")
        else:
            _clear_git_pull()
        runner : ScriptRunner = status.kwargs['runner']
        params : dict[str, Any] | None = status.kwargs.get('params')
        SC.click_script_runner_run(runner , params)