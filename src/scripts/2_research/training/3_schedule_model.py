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

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.res.api import ModelAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('schedule_model' , timeout = 10)
def main(**kwargs):
    schedule_name = kwargs.pop('schedule_name')
    short_test = kwargs.pop('short_test')
    resume = kwargs.pop('resume')
    resume = 1 if resume is None else resume * 1
    with AutoRunTask('train_schedule_model' , schedule_name , **kwargs) as runner:
        trainer = ModelAPI.schedule_model(schedule_name = schedule_name , short_test = short_test, resume = resume)
        runner.attach(trainer.result_package)
        runner.critical(f'Train schedule model at {runner.update_to} completed')
    return runner
        
if __name__ == '__main__':
    main()
