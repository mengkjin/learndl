"""
Quick call buttons for the interactive app, can be directly used in any page.
"""
from __future__ import annotations
import os
from src.api.interactive.util.quick_calls.basic import QuickCallButton

__all__ = [
    'Reboot' , 'TestLogger' , 'CheckCodeIssues' , "CheckDependencyVersion" , 'CheckConfigFiles' , 
    'ClearCatcherLogs' , 'ReplaceWeztermConfig' ,
    'CarryOutSchedules' , 'RebuildPreprocessedData' , 
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

class TestLogger(QuickCallButton):
    """Button that tests the streamlit app."""
    key = "test-logger"
    button_title = ".🩺 Logger"
    icon = ":material/developer_mode_tv:"
    default_help = 'Call Logger.test_logger() to test stdout.'
    color = 'cyan'
    
    def script_string(self) -> str:
        return """
            from src.proj import Logger
            Logger.test_logger()
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

class CheckDependencyVersion(QuickCallButton):
    """Button that checks the dependency version."""
    key = "check-dependency-version"
    button_title = "🔎 Dependencies"
    icon = ":material/package:"
    default_help = 'Check the dependency version in the project code if they are newer than the ones in pyproject.toml.'
    color = 'cyan'
    
    def script_string(self) -> str:
        return """
            from src.api.calls.source_code import CheckDependencyVersion
            CheckDependencyVersion.go()
        """

class ClearCatcherLogs(QuickCallButton):
    """Button that clears the catcher logs."""
    key = "clear-catcher-logs"
    button_title = "🆑 CatcherLog"
    icon = ":material/auto_delete:"
    default_help = 'Clear outdated catcher logs , default is 30 days.'
    color = 'red'
    
    def script_string(self) -> str:
        return """
            from src.api.calls.files import ClearOutdatedCatcherLogs
            ClearOutdatedCatcherLogs.go()
        """
class ReplaceWeztermConfig(QuickCallButton):
    """Button that replaces the wezterm config."""
    key = "replace-wezterm-config"
    button_title = "⚙️ Wezterm"
    icon = ":material/handyman:"
    default_help = 'Replace the wezterm config file by the project\'s default.'
    color = 'blue'
    
    def script_string(self) -> str:
        return """
            from src.api.calls.files import ReplaceWeztermConfig
            ReplaceWeztermConfig.go()
        """

class CarryOutSchedules(QuickCallButton):
    """Button that carries out the schedule model list."""
    key = "carry-out-schedules"
    button_title = "▶️ Schedules"
    icon = ":material/data_thresholding:"
    default_help = 'Carry out the schedule model list.'
    color = 'purple'
    research = True

    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        from src.api.calls.research import CarryOutScheduleModelList
        self.update(help = CarryOutScheduleModelList.get_description())
    
    def script_string(self) -> str:
        return """
            from src.api.calls.research import CarryOutScheduleModelList
            CarryOutScheduleModelList.go()
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

class CheckConfigFiles(QuickCallButton):
    """Button that modifies the config files."""
    key = "check-configs"
    button_title = "🔍 Config YAML"
    icon = ":material/stethoscope:"
    default_help = 'Check and auto modify the config files.'
    color = 'cyan'
    research = True
    
    def script_string(self) -> str:
        return """
            from src.api.calls.research import CheckAllConfigFiles
            CheckAllConfigFiles.go()
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