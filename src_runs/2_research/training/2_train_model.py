#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Train Model
# content: 训练某个新模型,模型的参数在configs/train/model.yaml里定义,也可以改变其他configs
# param_input: True
# param_placeholder: module_name
# email: True
# close_after_run: False

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask , MessageCapturer , PATH , Email
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    module = str(params['param'])
    with AutoRunTask('train model' , message_capturer = True , **params) as runner:
        trainer = ModelAPI.train_model(stage = 0 , resume = 0 , checkname= 1 , module = module if module else None)
        runner.add_attachments(trainer.path_training_output)

if __name__ == '__main__':
    main()
