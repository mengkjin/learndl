#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Model Predictions
# content: 更新所有Registered模型的预测结果

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    with AutoRunTask('update preds' , **argparse_dict()) as runner:
        ModelAPI.update_preds()

if __name__ == '__main__':
    main()