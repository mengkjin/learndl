#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-10-03
# description: Test Model
# content: 测试某个已训练的模型
# email: True
# mode: shell
# parameters:
#   mapping_name : 
#       type : "list(PATH.read_yaml(PATH.conf.joinpath('registry' , 'db_models_mapping')).keys())"
#       desc : choose a model
#       prefix : "db@"
#       required : True
#   start : 
#       type : int
#       desc : start date
#   end : 
#       type : int
#       desc : end date

from src.api import ModelAPI
from src.proj.util import ScriptTool

@ScriptTool('test_db_mapping' , '@mapping_name' , lock_num = 0)
def main(mapping_name : str | None = None , start : int | None = None , end : int | None = None , **kwargs):
    assert mapping_name is not None , 'mapping_name is required'
    ModelAPI.test_db_mapping(mapping_name , start = start , end = end)

if __name__ == '__main__':
    main()
