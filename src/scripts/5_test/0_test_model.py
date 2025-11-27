#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-06-20
# description: Test Model
# content: 测试某个已训练的模型
# email: True
# mode: shell
# parameters:
#   model_name : 
#       type : Options.available_models()
#       desc : choose a model
#       prefix : "model/"
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
#   start : 
#       type : int
#       desc : start date
#   end : 
#       type : int
#       desc : end date
# file_previewer:
#   name : "Model Output File Previewer"
#   path: "models/{model_name}/detailed_analysis/training_output.html"
#   height : 600 # optional

from src.api import ModelAPI
from src.app import ScriptTool

@ScriptTool('test_model' , '@model_name' , lock_num = 0)
def main(model_name : str | None = None , short_test : bool | None = None , start : int | None = None , end : int | None = None , **kwargs):
    assert model_name is not None , 'model_name is required'
    ModelAPI.test_model(model_name , short_test , start = start , end = end)

if __name__ == '__main__':
    main()
