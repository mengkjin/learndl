#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Change Power Saver
# content: 在Ubuntu系统中修改电源管理策略至"节能"
# email: False
# mode: os

from datetime import datetime
from src.proj import PATH
from src.app import change_power_mode , get_running_scripts

def main():
    running_scripts = get_running_scripts(__file__)
    log_path = PATH.log_main.joinpath('suspend','power_check.log')
    log_path.parent.mkdir(parents = True , exist_ok = True)
    if running_scripts:
        main_str = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} : Power set to saver'
        main_str += f' aborted due to running scripts: {running_scripts}\n'
        print(main_str , end = '')
        with open(log_path, 'a') as log_file:
            log_file.write(main_str)
    else:
        change_power_mode('power-saver' , log_path , True)

if __name__ == '__main__':
    main()