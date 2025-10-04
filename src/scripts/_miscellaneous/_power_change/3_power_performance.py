#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Change Power Performance
# content: 在Ubuntu系统中修改电源管理策略至"性能"
# email: False
# mode: os

from src.proj import PATH
from src.app import change_power_mode

def main():
    #running_scripts = get_running_scripts(__file__)
    log_path = PATH.log_main.joinpath('suspend','power_check.log')
    change_power_mode('performance' , log_path , True)

if __name__ == '__main__':
    main()