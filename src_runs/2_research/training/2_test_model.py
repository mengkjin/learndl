#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-06-20
# description: Test Model
# content: 测试某个已训练的模型
# email: True
# close_after_run: False
# param_inputs:
#   model_name : 
#       type : "[p.name for p in Path('models').iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]"
#       desc : choose a model
#       prefix : "model/"
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"

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
    with AutoRunTask('test model' , message_capturer = True , **kwargs) as runner:
        ModelAPI.test_model(model_name = runner['model_name'] , short_test = runner.get('short_test'))
        runner.critical(f'Test model at {runner.update_to} completed')

    return runner
        
if __name__ == '__main__':
    main()
