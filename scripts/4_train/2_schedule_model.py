#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-10-01
# description: Train Schedule Model
# content: 训练某个计划好的模型,模型的参数在configs/schedule或.local_resources/shared/schedule_model里定义,也可以改变configs
# email: True
# mode: shell
# parameters:
#   schedule_name : 
#       type : Options.available_schedules()
#       desc : schedule config file to train
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
#   resume : 
#       type : [True , False]
#       prefix : "resume/"
#       default : "resume/False"
#       desc : resume training , if change schedule file, set to False
#       required : True
#   start : 
#       type : int
#       desc : start date
#   end : 
#       type : int
#       desc : end date
# file_editor:
#   name : "Schedule Config File Editor"
#   path: "{PATH.conf}/schedule/{schedule_name}.yaml | {PATH.shared_schedule}/{schedule_name}.yaml"
#   height : 300 # optional

from src.api import ModelAPI
from src.proj.util import ScriptTool

@ScriptTool('train_schedule_model' , '@schedule_name' , lock_num = 2)
def main(schedule_name : str | None = None , short_test : bool | None = None , 
         resume : bool | None = None , start : int | None = None , end : int | None = None , **kwargs):
    assert schedule_name is not None , 'schedule_name is required'
    ModelAPI.schedule_model(schedule_name , short_test , 0 if resume is None else int(resume) , start = start , end = end)
        
if __name__ == '__main__':
    main()
