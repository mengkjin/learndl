#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-10-04
# description: Fix Past Data
# content: 修复历史数据,一事一议
# email: True
# mode: shell


from src.basic import ScriptTool
from src.basic import DB

@ScriptTool('fix_past_data')
def main(**kwargs):
    db_src = 'sellside'
    db_key = 'dongfang.scores_v0'
    dup_cols = ['secid' , 'date']
    for date in [20250801]:
        df = DB.load(db_src , db_key , date)
        if df.duplicated(dup_cols).any():
            DB.save(df.drop_duplicates(dup_cols , keep = 'last') , db_src , db_key , date)
    
if __name__ == '__main__':
    main()
