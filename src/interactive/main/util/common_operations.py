"""
define common operations for the interactive app to use.
"""
from __future__ import annotations
import streamlit as st
import subprocess
import time

from abc import abstractmethod , ABC
from dataclasses import dataclass
from src.proj import Proj , MACHINE , PATH , Const , BaseClass
from src.proj.util.options import Options
from src.interactive.backend import ScriptRunner
from .session_control import SC

@dataclass
class OperationStatus:
    """The status of the common operation."""
    disabled : bool = False
    help : str = ""

class CommonOperation(ABC , BaseClass.BoundLogger):
    """Abstract base for a single common operation.

    Subclasses define :attr:`key`, :attr:`icon`, and :attr:`title` as class
    variables and implement :meth:`button` to render the Streamlit widget.
    """
    key : str = ''
    title : str = ''
    icon : str = ''

    def __init__(self , **kwargs):
        self.kwargs = kwargs

    def __call__(self) -> None:
        """Call the common operation."""
        self.run()

    @property
    def status(self) -> OperationStatus:
        """The title for the common operation."""
        return OperationStatus()

    @abstractmethod
    def run(self) -> None:
        """Run the common operation."""
        ...

class RunCurrentScript(CommonOperation):
    """Submits the current script to the task queue."""
    key = f"script-runner-run"
    title = f"Run Script"
    icon = f":material/mode_off_on:"

    def update(self , **kwargs):
        self.kwargs = kwargs

    @property
    def runner(self) -> ScriptRunner | None:
        return self.kwargs.get('runner')

    @property
    def script_key(self) -> str | None:
        return self.runner.script_key if self.runner else None

    @property
    def status(self) -> OperationStatus:
        """The key for the common operation."""
        if self.runner is None or self.script_key is None:
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
        return OperationStatus(disabled , help_text)

    @property
    def help(self) -> str:
        """The help text for the common operation."""
        if self.script_key is None:
            return f"Please Choose a Script to Run First"
        if self.runner is None:
            return f"Please Choose a Runner to Run First"
        if SC.param_inputs_form is None:
            return f"Please Fill Required Parameters"
        if not SC.get_script_runner_validity(SC.param_inputs_form.param_values):
            return f"Please Fill Required Parameters"
        return f"Please Choose a Script to Run First" if self.script_key is None else f"Please Fill Required Parameters"

    @property
    def disabled(self) -> bool:
        """Whether the common operation is disabled."""
        return self.runner is None or self.script_key is None or SC.param_inputs_form is None or not SC.get_script_runner_validity(SC.param_inputs_form.param_values)

    def run(self) -> None:
        if self.disabled:
            return
        params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None
        SC.click_script_runner_run(self.runner , params)