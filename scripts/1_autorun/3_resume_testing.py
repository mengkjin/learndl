#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-12-31
# description: Resume Testing all prediction models and sellside/pooling factors
# content: Resume Testing all prediction models and sellside/pooling factors
# email: True
# mode: shell
# parameters:
#   force_resume : 
#       type : bool
#       desc : force resume testing even if it has been done in the last 24 hours
#       required : True
#       default : True

from src.api import ModelAPI
from src.proj import CALENDAR
from src.proj.util import ScriptTool

@ScriptTool('resume_testing' , CALENDAR.update_to() , forfeit_if_done = True)
def main(force_resume : bool = True , **kwargs):
    ModelAPI.resume_testing(force_resume = force_resume)

if __name__ == '__main__':
    main()