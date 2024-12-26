#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Suspend Machine
# content: 在Ubuntu系统中挂起系统，如果当前有运行脚本，则不挂起系统

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

import datetime , platform , subprocess
from pathlib import Path
from src_runs._abc import get_running_scripts

default_log_path = Path(__file__.removesuffix(__file__.split('learndl')[-1])).joinpath('logs','suspend','suspend_check.log')

def suspend_this_machine(log_path = default_log_path):
    running_scripts = get_running_scripts(__file__)
    do_suspend = not running_scripts and platform.system() != 'Windows'
    main_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path.parent.mkdir(parents = True , exist_ok = True)
    with open(log_path, 'a') as log_file:
        if running_scripts:
            main_str += f' : Suspension aborted due to running scripts: {running_scripts}\n'
        elif platform.system() == 'Windows':
            main_str += f' : Suspension aborted due windows platform\n'
        else:
            main_str += f' : Suspension applied\n'
        log_file.write(main_str)
        print(main_str , end = '')
    if do_suspend:
        subprocess.run(['systemctl', 'suspend'])

def main():
    suspend_this_machine()

if __name__ == '__main__':
    main()