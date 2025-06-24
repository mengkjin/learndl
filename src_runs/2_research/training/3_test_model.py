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
#       enum : [gru_day , gru_avg]
#       desc : choose a model
#       prefix : "model/"
#       required : True
#   short_test : 
#       enum : [True , False]
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
    model_name = params['model_name']
    short_test = eval(params.get('short_test' , 'None'))
    with AutoRunTask('test model' , message_capturer = True , **params) as runner:
        ModelAPI.test_model(model_name = model_name , short_test = short_test)
        
if __name__ == '__main__':
    main()
