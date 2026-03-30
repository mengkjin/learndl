#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj import DB
import pandas as pd
from src.proj.util import ScriptTool 
from src.data.crawler.announcement.agent import AnnouncementAgent

@ScriptTool('temporary_fix')
def main(**kwargs):
    for ex in ['bse' , 'sse' , 'szse']:
        paths = DB.paths('crawler' , f'announcement_{ex}')
        for path in paths:
            df = pd.read_feather(path)
            if 'ts_code' in df.columns:
                df = df.drop(columns = ['ts_code'])
            df['secid'] = DB.code2secid(df['sec_code'])
            assert df['secid'].dtype == int , f'path {path} secid is not int'
            DB.save_df(df , path)

    AnnouncementAgent.update()
        
if __name__ == '__main__':
    main()
        
    