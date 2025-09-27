#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Train Model
# content: 训练某个新模型,模型的参数在configs/train/model.yaml里定义,也可以改变其他configs
# email: True
# mode: shell
# parameters:
#   schedule_name : 
#       type : "[p.stem for p in Path('configs/schedule').glob('*.yaml')]"
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
# file_editor:
#   name : "Schedule Config File Editor"
#   path: "configs/schedule/{schedule_name}.yaml"
#   height : 300 # optional

from src.res.api import ModelAPI
from src.app.script_tool import ScriptTool

@ScriptTool('train_schedule_model' , '@schedule_name' , lock_num = 2)
def main(schedule_name : str | None = None , short_test : bool | None = None , resume : bool | None = None , **kwargs):
    assert schedule_name is not None , 'schedule_name is required'
    ModelAPI.schedule_model(schedule_name , short_test , 1 if resume is None else int(resume))
        
if __name__ == '__main__':
    main()
