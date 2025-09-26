#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Reconstruct Train Data
# content: 重建历史训练数据, 用于模型从2017年开始训练
# email: True
# mode: shell

from src.res.api import DataAPI
from src.app.script_tool import ScriptTool

@ScriptTool('reconstruct_train_data')
def main(**kwargs):
    DataAPI.reconstruct_train_data(confirm = 1)
    
if __name__ == '__main__':
    main()
