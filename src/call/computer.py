"""
Direct calls related to computer operations of this project.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from typing import Literal

from src.proj import MACHINE , Logger , PATH
from src.call.basic import DirectCall

__all__ = ['SuspendThisMachine' , 'ChangePowerMode' , 'PrintDiskSpaceInfo']

class SuspendThisMachine(DirectCall):
    """Suspend the machine if not windows and no running python scripts"""
    category = 'Computer'
    def run(self) -> None:
        import psutil
        import fnmatch
        running_scripts = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if not cmdline: 
                    continue
                for line in cmdline:
                    if any(fnmatch.fnmatch(line, pattern) for pattern in ['*.py' , '*.sh']) and line != __file__:
                        running_scripts.append(line)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        if running_scripts:
            Logger.critical(f'Suspension aborted due to running scripts: {set(running_scripts)}')
        elif MACHINE.is_windows:
            Logger.critical(f'Suspension aborted due Windows platform')
        elif MACHINE.is_macos:
            Logger.critical(f'Suspension aborted due MacOS platform')
        elif MACHINE.is_linux:
            Logger.critical(f'Suspension denied! Please manually suspend the machine.')
        else:
            Logger.critical(f'Suspension applied')
            subprocess.run(['systemctl', 'suspend'])

class ChangePowerMode(DirectCall):
    """Set Linux power profile via ``powerprofilesctl``; no-op log on Windows."""
    category = 'Computer'
    def __init__(self , mode : Literal['balanced' , 'power-saver' , 'performance'] | None = None , **kwargs):
        self.kwargs = kwargs | {'mode': mode}
    @property
    def mode(self) -> Literal['balanced' , 'power-saver' , 'performance'] | None:
        return self.kwargs['mode']
    @classmethod
    def get_description(cls , mode : Literal['balanced' , 'power-saver' , 'performance'] | None = None , **kwargs) -> str:
        if mode is None:
            return 'Power mode is not set, skip the operation.'
        return f'Set Linux power profile to {mode} via ``powerprofilesctl``; no-op log on Windows.'
    def run(self) -> None:
        if self.mode is None:
            Logger.critical(f'Power mode is not set')
            return
        main_str = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Power set to {self.mode}'
        if MACHINE.is_windows:
            main_str += f' aborted due windows platform\n'
        else:
            main_str += f' applied\n'
            subprocess.run(['powerprofilesctl', 'set', self.mode])
        Logger.stdout(main_str , end = '')

class PrintDiskSpaceInfo(DirectCall):
    """Human-readable disk usage summary for the project root."""
    category = 'Computer'
    
    def run(self):
        import shutil , os
        def format_bytes(bytes_num):
            """Format byte count with binary SI steps (1024)."""
            suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
            index = 0
            while bytes_num >= 1024 and index < len(suffixes) - 1:
                bytes_num /= 1024.0
                index += 1
            return f"{bytes_num:.2f} {suffixes[index]}"

        def get_disk_space_info():
            """Return dict of formatted total/used/free and usage percentages for ``PATH.main``."""
            total, used, free = shutil.disk_usage(PATH.main)
            percent_used = (used / total * 100) if total > 0 else 0
            percent_free = 100 - percent_used
            result = {
                'path': os.path.abspath(PATH.main),
                'total': format_bytes(total),
                'used': format_bytes(used),
                'free': format_bytes(free),
                'pct_used': round(percent_used, 2),
                'pct_free': round(percent_free, 2),
            }
            return result

        info = get_disk_space_info()
        Logger.stdout_pairs(info , title = 'Disk Space Info:')
        return info