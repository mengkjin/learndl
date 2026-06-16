#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2026-06-03
# description: Run Tracking Update
# content: 更新跟踪组合
# blacklist:
#   machine: ['Mathews-Mac']
# email: true
# mode: shell

from src.api.pkgs.update import UpdateAPI

from src.proj.util.script import ScriptTool

@ScriptTool('tracking_update')
def main(**kwargs):
    UpdateAPI.tracking_port()
        
if __name__ == '__main__':
    main()
    