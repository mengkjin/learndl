#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Weekly Update
# content: 每周更新模型(只在服务器上)
# email: True
# mode: shell

from src.res.api import ModelAPI
from src.proj import MACHINE
from src.basic import CALENDAR
from src.app.script_tool import ScriptTool

@ScriptTool('weekly_update' , CALENDAR.update_to() , forfeit_if_done = True)
def main(**kwargs):
    if not MACHINE.server:
        ScriptTool.error(f'{MACHINE.name} is not a server, skip weekly update')
    else:
        ModelAPI.update_models()


if __name__ == '__main__':
    main()