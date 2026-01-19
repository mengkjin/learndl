#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-01-18
# description: Unpack Model Files
# content: 解包模型文件
# email: True
# mode: shell


from src.proj.util import ScriptTool

from src.res.model.util import PredictionModel

@ScriptTool('unpack_model_files')
def main(**kwargs):
    PredictionModel.UnpackModelArchives()
    
if __name__ == '__main__':
    main()
