#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-10-04
# description: Fix Past Data
# content: 修复历史数据,一事一议
# email: True
# mode: shell


from src.app.script_tool import ScriptTool
from src.basic import DB

@ScriptTool('fix_past_data')
def main(**kwargs):
    df = DB.load(f'sellside' , 'dongfang.scores_v0' , 20240508).drop_duplicates()
    DB.save(df , f'sellside' , 'dongfang.scores_v0' , 20240508)
    
if __name__ == '__main__':
    main()
