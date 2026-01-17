import os
import shutil

from src.proj.env import PATH
from src.proj.log import Logger

__all__ = ['print_disk_space_info']

def format_bytes(bytes_num):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    index = 0
    while bytes_num >= 1024 and index < len(suffixes) - 1:
        bytes_num /= 1024.0
        index += 1
    return f"{bytes_num:.2f} {suffixes[index]}"

def get_disk_space_info():
    """get disk space info of a path"""
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
    info = get_disk_space_info()
    Logger.stdout("Disk Space Info:" , color = 'lightgreen' , bold = True)
    Logger.stdout_pairs(info , indent = 1 , color = 'lightgreen' , bold = True)
    return info