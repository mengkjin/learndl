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
#       type : str
#       desc : module to train
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    module = params.pop('module_name')
    short_test = eval(params.pop('short_test' , 'None'))
    with AutoRunTask('train model' , **params) as runner:
        ModelAPI.train_model(module = module if module else None , short_test = short_test)
        
if __name__ == '__main__':
    main()
