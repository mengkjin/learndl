#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Suspend Machine
# content: 在Ubuntu系统中挂起系统，如果当前有运行脚本，则不挂起系统


import platform , subprocess
from datetime import datetime
from src.proj import PATH
from src.app import get_running_scripts

default_log_path = PATH.log_main.joinpath('suspend','suspend_check.log')

def suspend_this_machine(log_path = default_log_path):
    running_scripts = get_running_scripts(__file__)
    do_suspend = not running_scripts and platform.system() != 'Windows'
    log_path.parent.mkdir(parents = True , exist_ok = True)
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as log_file:
        if running_scripts:
            main_str = f'{time_str} : Suspension aborted due to running scripts: {running_scripts}\n'
        elif platform.system() == 'Windows':
            main_str = f'{time_str} : Suspension aborted due windows platform\n'
        else:
            main_str = f'{time_str} : Suspension applied\n'
        log_file.write(main_str)
        print(main_str , end = '')
    if do_suspend:
        subprocess.run(['systemctl', 'suspend'])

def main():
    suspend_this_machine()

if __name__ == '__main__':
    main()