#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-05-09
# description: modify model configs
# content: modify model configs
# email: True
# mode: shell

from src.proj.util.script import ScriptTool
from src.res.model.util.config import check_all_config_files

@ScriptTool('config_modifier')
def main(**kwargs):     
    check_all_config_files()

if __name__ == '__main__':
    main()