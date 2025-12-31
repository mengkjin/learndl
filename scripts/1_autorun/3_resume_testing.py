#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-12-31
# description: Resume Testing all prediction models and sellside/pooling factors
# content: Resume Testing all prediction models and sellside/pooling factors
# email: True
# mode: shell

from src.api import UpdateAPI
from src.proj import CALENDAR
from src.proj.util import ScriptTool

@ScriptTool('resume_testing' , CALENDAR.update_to() , forfeit_if_done = True)
def main(**kwargs):
    UpdateAPI.resume_testing()

if __name__ == '__main__':
    main()