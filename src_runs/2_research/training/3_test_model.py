#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-06-20
# description: Test Model
# content: 测试某个已训练的模型
# param_input: True
# param_placeholder: model_name
# email: True
# close_after_run: False

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    model_name = str(params['param'])
    assert model_name , f'Please input model name'
    with AutoRunTask('test model' , message_capturer = True , **params) as runner:
        ModelAPI.test_model(model_name = model_name)
        
if __name__ == '__main__':
    main()
