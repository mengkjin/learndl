#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Train Model
# content: 训练某个新模型,模型的参数在configs/train/model.yaml里定义,也可以改变其他configs
# email: True
# mode: shell
# parameters:
#   module_name : 
#       type : "Path('.local_resources/temp/available_modules.txt').read_text().splitlines()"
#       desc : module to train
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
# file_editor:
#   name : "Model Config File Editor"
#   path: "configs/{module_name}.yaml"
#   height : 300 # optional

from pathlib import Path
from src.res.api import ModelAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('train_model' , timeout = 10)
def main(**kwargs):
    module_name = kwargs.pop('module_name')
    module = Path(module_name).parts[-1]
    with AutoRunTask('train_model' , module , **kwargs) as runner:
        trainer = ModelAPI.train_model(module = module , short_test = runner.get('short_test'))
        runner.attach(trainer.result_package)
        runner.critical(f'Train model at {runner.update_to} completed')
    return runner
        
if __name__ == '__main__':
    main()
