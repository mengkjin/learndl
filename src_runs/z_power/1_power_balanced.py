#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Change Power Balanced
# content: 在Ubuntu系统中修改电源管理策略至"平衡"
# email: False
# close_after_run: True

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

import datetime
from pathlib import Path
from src_runs._abc import change_power_mode , get_running_scripts

def main():
    #running_scripts = get_running_scripts(__file__)
    log_path = paths[0].joinpath('logs','suspend','power_check.log')
    change_power_mode('balanced' , log_path , True)

if __name__ == '__main__':
    main()