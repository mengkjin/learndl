#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-10-08
# description: Recalculate All Pred
# content: 重新计算所有预测数据
# email: True
# mode: shell
# parameters:
#   start : 
#       type : int
#       desc : start yyyymmdd
#       min : 20170101
#       max : 99991231
#       required : True
#       default : 20170101
#   end : 
#       type : int
#       desc : end yyyymmdd
#       min : 20170101
#       max : 99991231
#       required : True
#       default : 20241231


from src.app.script_tool import ScriptTool
from src.res.api import ModelAPI

@ScriptTool('recalc_preds')
def main(start : int | None = None , end : int | None = None , **kwargs):
    ModelAPI.recalculate_preds(start_dt = start , end_dt = end)
    
if __name__ == '__main__':
    main()
