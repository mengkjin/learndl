#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 
from src.proj import DB 
from src.res.factor.risk import TuShareCNE5_Calculator

@ScriptTool('temporary_fix')
def main(**kwargs):
    calculator = TuShareCNE5_Calculator()
    paths = DB.paths('models' , 'tushare_cne5_cov' , start = 0)
    for path in paths:
        df = DB.load_df(path)
        if 'factor_name' in df.columns:
            continue
        if df.columns.to_list()[0] == 'index':
            df = df.rename(columns={'index':'factor_name'})
        else:
            date = DB.path_date(path)
            df = calculator.calc_common_risk(date)
        DB.save_df(df , path)
        
if __name__ == '__main__':
    main()
        
    