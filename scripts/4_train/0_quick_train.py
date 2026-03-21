#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-03-20
# description: Quick Train Model
# content: 基于默认的模型配置,快速训练(short_test = True)一个模型
# email: False
# mode: shell
# file_editor:
#   name : "Model Config File Editor"
#   path: "configs/model/model.yaml"
#   height : 300 # optional

from src.api import ModelAPI
from src.proj.util import ScriptTool

@ScriptTool('quick_train_model')
def main(**kwargs):
    ModelAPI.train_model(short_test = True)
        
if __name__ == '__main__':
    main()
