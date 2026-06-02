"""
define common operations for the interactive app to use.
"""
from __future__ import annotations
import streamlit as st
import subprocess
import time

from abc import abstractmethod , ABC
from dataclasses import dataclass
from functools import cached_property
from src.proj import Proj , MACHINE , PATH , Const , BaseClass
from src.proj.util.options import Options
from src.interactive.backend import ScriptRunner
from .session_control import SC

@dataclass
class OperationStatus:
    """The status of the common operation."""
    disabled : bool = True
    help : str = ""

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

    @cached_property
    def status(self) -> OperationStatus:
        """The title for the common operation."""
        return OperationStatus()

    @abstractmethod
    def run(self) -> None:
        """Run the common operation."""
        ...

class RunCurrentScript(CommonOperation):
    """Submits the current script to the task queue."""
    def update_kwargs(self , **kwargs):
        self.kwargs = kwargs
        self.__dict__.pop('status', None)

    @property
    def runner(self) -> ScriptRunner | None:
        return self.kwargs.get('runner')

    @cached_property
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
        return OperationStatus(disabled , help_text)

    def run(self) -> None:
        if self.status.disabled:
            return
        params = SC.param_inputs_form.param_values if SC.param_inputs_form is not None else None
        SC.click_script_runner_run(self.runner , params)