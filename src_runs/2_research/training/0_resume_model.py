#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-08-03
# description: Resume Model
# content: 恢复某个已训练的模型继续训练
# email: True
# close_after_run: False
# param_inputs:
#   model_name : 
#       type : "[p.name for p in Path('models').iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]"
#       desc : choose a model
#       prefix : "model/"
#       required : True
# file_previewer:
#   name : "Model Output File Previewer"
#   path: "models/{model_name}/detailed_analysis/training_output.html"
#   height : 600 # optional

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_app import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    with AutoRunTask('resume model' , **kwargs) as runner:
        ModelAPI.resume_model(model_name = runner['model_name'])
        runner.critical(f'Resume model at {runner.update_to} completed')

    return runner
        
if __name__ == '__main__':
    main()
