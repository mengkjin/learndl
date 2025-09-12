#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-06-20
# description: Test Model
# content: 测试某个已训练的模型
# email: True
# mode: shell
# parameters:
#   model_name : 
#       type : "[p.name for p in Path('models').iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]"
#       desc : choose a model
#       prefix : "model/"
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
# file_previewer:
#   name : "Model Output File Previewer"
#   path: "models/{model_name}/detailed_analysis/training_output.html"
#   height : 600 # optional

from src.res.api import ModelAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('test_model' , timeout = 10)
def main(**kwargs):
    model_name = kwargs.pop('model_name')
    with AutoRunTask('test_model' , model_name , **kwargs) as runner:
        ModelAPI.test_model(model_name = model_name , short_test = runner.get('short_test'))
        runner.critical(f'Test model at {runner.update_to} completed')

    return runner
        
if __name__ == '__main__':
    main()
