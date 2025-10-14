#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Model Predictions
# content: 更新所有Registered模型的预测结果
# email: True
# mode: shell

from src.res.api import ModelAPI
from src.app.script_tool import ScriptTool

@ScriptTool('update_preds')
def main(**kwargs):
    ModelAPI.update_preds() 

if __name__ == '__main__':
    main()
