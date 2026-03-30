"""Human-readable disk usage summary for the project root."""

import os
import shutil

from src.proj.env import PATH
from src.proj.log import Logger

__all__ = ['print_disk_space_info']

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

def print_disk_space_info():
    """Show disk space info in the best way."""
    info = get_disk_space_info()
    Logger.stdout_pairs(info , title = 'Disk Space Info:')
    return info