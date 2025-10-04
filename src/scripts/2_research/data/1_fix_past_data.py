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
    dates = DB.dates('sellside' , 'dongfang.scores_v0')
    for date in [20250801]:
        df = DB.load(f'sellside' , 'dongfang.scores_v0' , date)
        print(df.drop_duplicates().any())
        DB.save(df.drop_duplicates() , f'sellside' , 'dongfang.scores_v0' , date)
    
if __name__ == '__main__':
    main()
