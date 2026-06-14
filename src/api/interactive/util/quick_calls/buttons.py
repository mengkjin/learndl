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
    'ArchiveCurrentModel' , 'ResumeArchivedModel' , 
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
    icon = ":material/keyboard_external_input:"
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
    icon = ":material/troubleshoot:"
    default_help = 'Check the code issues in the project code.'
    color = 'cyan'
    
    def script_string(self) -> str:
        return """
            from src.api.calls.files import CheckCodeIssues
            CheckCodeIssues.go()
        """

class CheckDependencyVersion(QuickCallButton):
    """Button that checks the dependency version."""
    key = "check-dependency-version"
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
    icon = ":material/component_exchange:"
    default_help = 'Resume models from model archive directory, you can choose which model to resume.'
    color = 'blue'
    
    def script_string(self) -> str:
        return """
            from src.api.calls.files import ReplaceWeztermConfig
            ReplaceWeztermConfig.go()
        """

class CarryOutSchedules(QuickCallButton):
    """Button that carries out the schedule model list."""
    key = "carry-out-schedules"
    icon = ":material/order_play:"
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

class CheckConfigFiles(QuickCallButton):
    """Button that modifies the config files."""
    key = "check-configs"
    icon = ":material/search_insights:"
    default_help = 'Check and auto modify the config files.'
    color = 'cyan'
    research = True
    
    def script_string(self) -> str:
        return """
            from src.api.calls.research import CheckAllConfigFiles
            CheckAllConfigFiles.go()
        """

class RebuildPreprocessedData(QuickCallButton):
    """Button that rebuilds the preprocessed data."""
    key = "rebuild-preprocess"
    icon = ":material/calculate:"
    default_help = 'Rebuild the preprocessed data, you can choose which data and which type to rebuild.'
    color = 'purple'
    research = True
    
    def script_string(self) -> str:
        return """
            from src.api.calls.data import ReconstructPreprocessedData
            ReconstructPreprocessedData.go()
        """

class ArchiveCurrentModel(QuickCallButton):
    """Button that archives the model."""
    key = "archive-current-model"
    icon = ":material/archive:"
    default_help = 'Archive models from model directory to model archive directory, you can choose which model to archive.'
    color = 'pink'
    research = True

    def script_string(self) -> str:
        return """
            from src.api.calls.files import ArchiveCurrentModel
            ArchiveCurrentModel.go()
        """

class ResumeArchivedModel(QuickCallButton):
    """Button that resumes the model."""
    key = "resume-archived-model"
    icon = ":material/unarchive:"
    default_help = 'Resume models from model archive directory, you can choose which model to resume.'
    color = 'green'
    research = True
    
    def script_string(self) -> str:
        return """
            from src.api.calls.files import ResumeArchivedModel
            ResumeArchivedModel.go()
        """

class Tensorboard(QuickCallButton):
    """Button that launches Tensorboard."""
    key = "tensorboard"
    icon = ":material/network_intel_node:"
    default_help = 'Launch Tensorboard. You can choose which option to launch.'
    color = 'gold'
    research = True

    def script_string(self) -> str:
        return """
            from src.api.pkgs import DashboardAPI
            DashboardAPI.tensorboard()
        """

class OptunaDashboard(QuickCallButton):
    """Button that launches Optuna Dashboard."""
    key = "optuna"
    icon = ":material/target:"
    default_help = 'Launch Optuna Dashboard. You can choose which option to launch.'
    color = 'gold'
    research = True

    def script_string(self) -> str:
        return """
            from src.api.pkgs import DashboardAPI
            DashboardAPI.optuna_dashboard()
        """