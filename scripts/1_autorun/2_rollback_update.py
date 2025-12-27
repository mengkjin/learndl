#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-08-18
# description: Run Rollback Update
# content: 回滚Tushare更新数据和因子，不重新更新其他数据(当做已经发生)
# email: True
# mode: shell
# parameters:
#   rollback_date : 
#       type : int
#       desc : from which date to rollback and update subsequent data (update start at +1 date)
#       required : True
#       default : 20250814

from src.api import UpdateAPI
from src.proj.util import ScriptTool

@ScriptTool('rollback_update' , '@rollback_date')
def main(rollback_date : int | None = None , **kwargs):
    assert rollback_date is not None , 'rollback_date is required'
    UpdateAPI.rollback(rollback_date = rollback_date)
        
if __name__ == '__main__':
    main()
        
    