#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Change Power Performance
# content: 在Ubuntu系统中修改电源管理策略至"性能"

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

import datetime , fnmatch , platform , psutil , subprocess
from pathlib import Path

def get_running_scripts(exclude_scripts : list[str] = [] , script_type = ['*.py', '*.sh'] , default_excludes = ['kernel_interrupt_daemon.py']):
    running_scripts : list[Path] = []
    self = Path(__file__)
    exclude_scripts = exclude_scripts + default_excludes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if not cmdline: continue
            for line in cmdline:
                if any(fnmatch.fnmatch(line, pattern) for pattern in script_type):
                    script = Path(line)
                    if script == self or any(scp in line for scp in exclude_scripts): 
                        pass
                    else:
                        running_scripts.append(Path(line))
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return running_scripts

def change_power_performance(exclude_scripts : list[str] = [] , log_path = paths[0].joinpath('logs','suspend','power_check.log')):
    running_scripts = get_running_scripts(exclude_scripts)
    do_change = not running_scripts and platform.system() != 'Windows'
    main_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' : Power set to performance'
    log_path.parent.mkdir(parents = True , exist_ok = True)
    with open(log_path, 'a') as log_file:
        if platform.system() == 'Windows':
            main_str += f' aborted due windows platform\n'
        else:
            main_str += f' applied\n'
        log_file.write(main_str)
        print(main_str , end = '')
    if do_change:
        subprocess.run(['powerprofilesctl', 'set', 'performance'])

def main():
    change_power_performance(exclude_scripts = ['3_power_performance.py'])

if __name__ == '__main__':
    main()