#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Hiddens Extraction
# content: 更新模型隐变量,用于做其他模型的输入
# email: True
# mode: shell

from src.api import ModelAPI
from src.proj.util import ScriptTool

@ScriptTool('update_hiddens')
def main(**kwargs):
    ModelAPI.update_hidden()
    
if __name__ == '__main__':
    main()
