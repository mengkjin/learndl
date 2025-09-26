#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update AI Models
# content: 更新训练所有Registered模型
# email: True
# mode: shell


from src.res.api import ModelAPI
from src.app.script_tool import ScriptTool

@ScriptTool('update_models')
def main(**kwargs):
    ModelAPI.update_models()

if __name__ == '__main__':
    main()
