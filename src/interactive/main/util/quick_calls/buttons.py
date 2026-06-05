"""
Quick call buttons for the interactive app, can be directly used in any page.
"""
from __future__ import annotations
from src.interactive.main.util.quick_calls.basic import QuickCallButton

class Reboot(QuickCallButton):
    """Button that reboots the streamlit server."""
    key = "reboot"
    icon = ":material/lightning_stand:"
    default_help = 'Reboot the streamlit server, will kill the current streamlit server and restart a new one.'
    color = 'purple'
    done_action = 'close'

    def script_string(self) -> str:
        import os
        current_pid = os.getpid()
        return f"""
            from src.call.app import kill_and_reboot_app
            kill_and_reboot_app({current_pid})
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

class CheckConfigFiles(QuickCallButton):
    """Button that modifies the config files."""
    key = "check-configs"
    icon = ":material/search_insights:"
    default_help = 'Check and auto modify the config files.'
    color = 'cyan'

    def script_string(self) -> str:
        return """
            from src.call.files import check_all_config_files
            check_all_config_files()
        """

class RebuildPreprocessedData(QuickCallButton):
    """Button that rebuilds the preprocessed data."""
    key = "rebuild-preprocess"
    icon = ":material/calculate:"
    default_help = 'Rebuild the preprocessed data, you can choose which data and which type to rebuild.'
    color = 'gold'
    
    def script_string(self) -> str:
        return """
            from src.call.data import rebuild_preprocessed_data
            rebuild_preprocessed_data()
        """

class Tensorboard(QuickCallButton):
    """Button that launches Tensorboard."""
    key = "tensorboard"
    icon = ":material/network_intel_node:"
    default_help = 'Launch Tensorboard. You can choose which option to launch.'
    color = 'gold'

    def script_string(self) -> str:
        return """
            from src.api import DashboardAPI
            DashboardAPI.tensorboard()
        """

class OptunaDashboard(QuickCallButton):
    """Button that launches Optuna Dashboard."""
    key = "optuna"
    icon = ":material/target:"
    default_help = 'Launch Optuna Dashboard. You can choose which option to launch.'
    color = 'gold'

    def script_string(self) -> str:
        return """
            from src.api import DashboardAPI
            DashboardAPI.optuna_dashboard()
        """

class ClearCatcherLogs(QuickCallButton):
    """Button that clears the catcher logs."""
    key = "clear-catcher-logs"
    icon = ":material/auto_delete:"
    default_help = 'Clear outdated catcher logs , default is 30 days.'
    color = 'red'
    
    def script_string(self) -> str:
        return """
            from src.call.files import clear_outdated_catcher_logs
            clear_outdated_catcher_logs()
        """
class ArchiveModel(QuickCallButton):
    """Button that archives the model."""
    key = "archive-model"
    icon = ":material/archive:"
    default_help = 'Archive models from model directory to model archive directory, you can choose which model to archive.'
    color = 'pink'
    
    def script_string(self) -> str:
        return """
            from src.call.files import archive_current_model
            archive_current_model()
        """

class ResumeModel(QuickCallButton):
    """Button that resumes the model."""
    key = "resume-model"
    icon = ":material/unarchive:"
    default_help = 'Resume models from model archive directory, you can choose which model to resume.'
    color = 'green'
    
    def script_string(self) -> str:
        return """
            from src.call.files import resume_archived_model
            resume_archived_model()
        """
class ReplaceWeztermConfig(QuickCallButton):
    """Button that replaces the wezterm config."""
    key = "replace-wezterm-config"
    icon = ":material/component_exchange:"
    default_help = 'Resume models from model archive directory, you can choose which model to resume.'
    color = 'blue'
    
    def script_string(self) -> str:
        return """
            from src.call.files import replace_wezterm_config
            replace_wezterm_config()
        """