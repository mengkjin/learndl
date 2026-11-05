#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2025-12-31
# description: Resume Testing all prediction models and sellside/pooling factors
# content: Resume Testing all prediction models and sellside/pooling factors
# email: True
# mode: shell

from src.proj import CALENDAR
from src.proj.util.script import ScriptTool

@ScriptTool('resume_testing' , CALENDAR.update_to() , forfeit_if_done = True)
def main(**kwargs):
    from src.api.pkgs.model import ModelAPI
    from src.api.pkgs.summary import SummaryAPI
    ModelAPI.resume_testing(force_resume = True)
    SummaryAPI.update()

if __name__ == '__main__':
    main()