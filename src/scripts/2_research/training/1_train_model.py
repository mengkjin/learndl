#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Train Model
# content: 训练某个新模型,模型的参数在configs/train/model.yaml里定义,也可以改变其他configs
# email: True
# mode: shell
# parameters:
#   module_name : 
#       type : "Path('.local_resources/temp/available_modules.txt').read_text().splitlines()"
#       desc : module to train
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
# file_editor:
#   name : "Model Config File Editor"
#   path: "configs/{module_name}.yaml"
#   height : 300 # optional

from pathlib import Path
from src.res.api import ModelAPI
from src.app.script_tool import ScriptTool

@ScriptTool('train_model' , '@module_name')
def main(module_name : str | None = None , short_test : bool | None = None , **kwargs):
    assert module_name is not None , 'module_name is required'
    ModelAPI.train_model(Path(module_name).parts[-1] , short_test)
        
if __name__ == '__main__':
    main()
