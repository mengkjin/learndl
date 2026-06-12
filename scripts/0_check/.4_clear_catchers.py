#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-06-02
# description: clear catcher logs
# content: clear catcher logs
# email: True
# mode: shell
# parameters:
#   days_ago : 
#       type : int
#       desc : clear catcher logs that are older than days_ago days
#       default : 30

from src.proj.util.script import ScriptTool
from src.call.files import ClearOutdatedCatcherLogs

@ScriptTool('clear_catchers')
def main(days_ago : int = 30 , **kwargs): 
    ClearOutdatedCatcherLogs.go(days_ago=days_ago)
    
if __name__ == '__main__':
    main()