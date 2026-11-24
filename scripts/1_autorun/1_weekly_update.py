#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2024-11-27
# description: Run Weekly Update
# content: 每周维护训练分析数据并增量更新 fit 数据
# blacklist:
#   machine: ['Mathews-Mac']
# email: True
# mode: shell

from src.proj import CALENDAR
from src.proj.util.script import ScriptTool

@ScriptTool('weekly_update' , CALENDAR.update_to() , forfeit_if_done = True)
def main(**kwargs):
    from src.api.pkgs.update import UpdateAPI
    UpdateAPI.weekly()

if __name__ == '__main__':
    main()