#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-01-18
# description: Pack Model Files
# content: 打包模型文件
# email: True
# mode: shell
# parameters:
#   start_model_date : 
#       type : int
#       desc : start model date
#       required : True
#       default : 20240101

from src.proj.util import ScriptTool

from src.res.model.util import PredictionModel

@ScriptTool('pack_model_files')
def main(start_model_date : int = 20240101 , **kwargs):
    PredictionModel.PackModelArchives(start_model_date = start_model_date)
    
if __name__ == '__main__':
    main()
