#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-08-03
# description: Resume Model
# content: 恢复某个已训练的模型继续训练
# email: True
# mode: shell
# parameters:
#   model_name : 
#       type : "[p.name for p in Path('models').iterdir() if not p.name.endswith('_ShortTest') and not p.name.startswith('.')]"
#       desc : choose a model
#       prefix : "model/"
#       required : True
# file_previewer:
#   name : "Model Output File Previewer"
#   path: "models/{model_name}/detailed_analysis/training_output.html"
#   height : 600 # optional

from src.res.api import ModelAPI
from src.app.script_tool import ScriptTool

@ScriptTool('resume_model' , '@model_name')
def main(model_name : str | None = None , **kwargs):
    assert model_name is not None , 'model_name is required'
    ModelAPI.resume_model(model_name = model_name)
        
if __name__ == '__main__':
    main()
