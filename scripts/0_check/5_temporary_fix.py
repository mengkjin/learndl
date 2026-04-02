#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 
from src.api import UpdateAPI #
from src.data.crawler.announcement.agent import AnnouncementAgent #
from src.data.util.classes import DataCache
from src.proj import DB , MACHINE
from src.res.factor.risk import TuShareCNE5_Calculator

@ScriptTool('temporary_fix')
def main(**kwargs):
    if MACHINE.is_windows:
        DataCache.purge_all(confirm = True)
    # AnnouncementAgent.update()
    calculator = TuShareCNE5_Calculator()
    paths = DB.paths('models' , 'tushare_cne5_coef' , start = 0)
    for path in paths:
        df = DB.load_df(path)
        if 'factor_name' in df.columns:
            ...
        if df.index.name == 'factor_name':
            df = df.reset_index(drop = False)
        elif df.index.name is None and df.index.get_level_values(None)[0] == 'market': # type: ignore
            df.index.name = 'factor_name'
            df = df.reset_index(drop = False)
        else:
            date = DB.path_date(path)
            df = calculator.get_coef(date , read = False)
        DB.save_df(df , path)
        
if __name__ == '__main__':
    main()
        
    