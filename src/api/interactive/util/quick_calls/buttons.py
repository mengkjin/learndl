"""
Quick call buttons for the interactive app, can be directly used in any page.
"""
from __future__ import annotations
import os
from src.api.interactive.util.quick_calls.basic import QuickCallButton

__all__ = [
    'Reboot' , 'TestCode' , 'CheckCodeIssues' , 
    'ProjectAutoFix' ,
    'CarryOutScheduleWorkList' , 'RebuildPreprocessedData' , 
    'ModelArchiveOperations' ,
    'Tensorboard' , 'OptunaDashboard' , 
]

class Reboot(QuickCallButton):
    """Button that reboots the streamlit server."""
    key = "reboot"
    icon = ":material/lightning_stand:"
    default_help = 'Reboot the streamlit server, will kill the current streamlit server and restart a new one.'
    color = 'purple'
    done_action = 'close'

    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        # dynamic help string based on the current pid and calling process
        from src.api.calls.app import KillAndRebootApp
        self.current_pid = os.getpid()
        self.update(help = KillAndRebootApp.get_description(running_pid=self.current_pid))

    def script_string(self) -> str:
        return f"""
            from src.api.calls.app import KillAndRebootApp
            KillAndRebootApp.go(running_pid={self.current_pid})
        """

class TestCode(QuickCallButton):
    """Button that tests the project code."""
    key = "test-project-code"
    button_title = ".🩺 Test Code"
    icon = ":material/developer_mode_tv:"
    default_help = 'Running tests for the project code, including logger , quick train , parallel factor calculation.'
    color = 'orange'
    
    def script_string(self) -> str:
        return """
            from src.api.calls.test import TestCode
            TestCode.go()
        """

class CheckCodeIssues(QuickCallButton):
    """Button that checks the code issues."""
    key = "check-code-issues"
    button_title = "🔎 Code Issues"
    icon = ":material/troubleshoot:"
    default_help = 'Check the code issues in the project code.'
    color = 'cyan'
    
    def script_string(self) -> str:
        return """
            from src.api.calls.source_code import CheckCodeIssues
            CheckCodeIssues.go()
        """

class ProjectAutoFix(QuickCallButton):
    """Button that applies the project patches."""
    key = "project-auto-fix"
    button_title = "🔧 AutoFix"
    icon = ":material/handyman:"
    default_help = 'Apply the project auto fixes , including check & fix all config files , replace wezterm config , clear outdated catcher logs.'
    color = 'blue'
    
    def script_string(self) -> str:
        return """
            from src.api.calls.files import ProjectAutoFix
            ProjectAutoFix.go()
        """

class CarryOutScheduleWorkList(QuickCallButton):
    """Button that carries out the schedule model list."""
    key = "carry-out-schedules"
    button_title = "▶️ Schedules"
    icon = ":material/data_thresholding:"
    default_help = 'Carry out the schedule model work list.'
    color = 'purple'
    research = True

    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        from src.api.calls.research import CarryOutScheduleWorkList
        self.update(help = CarryOutScheduleWorkList.get_description())
    
    def script_string(self) -> str:
        return """
            from src.api.calls.research import CarryOutScheduleWorkList
            CarryOutScheduleWorkList.go()
        """

class RebuildPreprocessedData(QuickCallButton):
    """Button that rebuilds the preprocessed data."""
    key = "rebuild-preprocess"
    button_title = f"🧱 PreProcess"
    icon = ":material/calculate:"
    default_help = 'Rebuild the preprocessed data, you can choose which data and which type to rebuild.'
    color = 'purple'
    research = True
    
    def script_string(self) -> str:
        return """
            from src.api.calls.data import ReconstructPreprocessedData
            ReconstructPreprocessedData.go()
        """

class ModelArchiveOperations(QuickCallButton):
    """Button that manages the model archive operations."""
    key = "model-archive-operations"
    button_title = "📂 Models"
    icon = ":material/home_storage_gear:"
    default_help = 'Manage the model archive operations, including archive / resume / rename / packing.'
    color = 'pink'
    research = True
    
    def script_string(self) -> str:
        return """
            from src.api.calls.files import ModelArchiveOperations
            ModelArchiveOperations.go()
        """

class Tensorboard(QuickCallButton):
    """Button that launches Tensorboard."""
    key = "tensorboard"
    button_title = "TensorBoard"
    icon = ":material/network_intel_node:"
    default_help = 'Launch Tensorboard. You can choose which option to launch.'
    color = 'gold'
    research = True

    def script_string(self) -> str:
        return """
            from src.api.pkgs.dashboard import DashboardAPI
            DashboardAPI.tensorboard()
        """

class OptunaDashboard(QuickCallButton):
    """Button that launches Optuna Dashboard."""
    key = "optuna"
    button_title = "Optuna Dashboard"
    icon = ":material/target:"
    default_help = 'Launch Optuna Dashboard. You can choose which option to launch.'
    color = 'gold'
    research = True

    def script_string(self) -> str:
        return """
            from src.api.pkgs.dashboard import DashboardAPI
            DashboardAPI.optuna_dashboard()
        """