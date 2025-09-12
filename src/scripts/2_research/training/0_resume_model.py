#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-08-03
# description: Resume Model
# content: 恢复某个已训练的模型继续训练
# email: True
# mode: shell
# parameters:
#   model_name : 
#       type : "[p.name for p in Path('models').iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]"
#       desc : choose a model
#       prefix : "model/"
#       required : True
# file_previewer:
#   name : "Model Output File Previewer"
#   path: "models/{model_name}/detailed_analysis/training_output.html"
#   height : 600 # optional

from src.res.api import ModelAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('resume_model' , timeout = 10)
def main(**kwargs):
    model_name = kwargs.pop('model_name')
    with AutoRunTask('resume_model' , model_name , **kwargs) as runner:
        ModelAPI.resume_model(model_name = model_name)
        runner.critical(f'Resume model at {runner.update_to} completed')

    return runner
        
if __name__ == '__main__':
    main()
