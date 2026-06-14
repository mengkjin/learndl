#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-05-09
# description: modify model configs
# content: modify model configs
# email: True
# mode: shell

from src.proj.util.script import ScriptTool
from src.api.calls.research import CheckAllConfigFiles

@ScriptTool('config_modifier')
def main(**kwargs):     
    CheckAllConfigFiles.go()

if __name__ == '__main__':
    main()