#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Weekly Update
# content: 每周更新模型(只在服务器上)
# email: True
# mode: shell

from src.api import UpdateAPI
from src.basic import CALENDAR
from src.basic import ScriptTool

@ScriptTool('weekly_update' , CALENDAR.update_to() , forfeit_if_done = True)
def main(**kwargs):
    UpdateAPI.weekly()

if __name__ == '__main__':
    main()