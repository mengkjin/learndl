#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Train Model
# content: 训练某个新模型,模型的参数在configs/train/model.yaml里定义,也可以改变其他configs
# email: True
# close_after_run: False
# param_inputs:
#   module_name : 
#       type : "Path('available_modules.txt').read_text().splitlines()"
#       desc : module to train
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
# file_editor:
#   path: "configs/{module_name}.yaml"

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_ui import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    with AutoRunTask('train model' , **kwargs) as runner:
        module = runner.get('module_name').split('/')[-1]
        trainer = ModelAPI.train_model(module = module , short_test = runner.get('short_test'))
        runner.attach(trainer.result_package)
        runner.critical(f'Train model at {runner.update_to} completed')
    return runner
        
if __name__ == '__main__':
    main()
