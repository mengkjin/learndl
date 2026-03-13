#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 

@ScriptTool('temporary_fix')
def main(**kwargs):
    from src.proj import DB , CALENDAR
    dates = CALENDAR.td_within(20170101 , 20991231)
    for date in dates:
        df = DB.load('pred' , 'gru_day_V0' , date)
        if not df.empty and 'gru_day_V0' not in df.columns:
            df = df.rename(columns={'gru_day': 'gru_day_V0'})
            assert 'gru_day_V0' in df.columns , f'gru_day_V0 not in df.columns: {df.columns}'
            DB.save(df , 'pred' , 'gru_day_V0' , date)

        df = DB.load('pred' , 'gru_day_V1' , date)
        if not df.empty and 'gru_day_V1' not in df.columns  :
            df = df.rename(columns={'gru_avg': 'gru_day_V1'})
            assert 'gru_day_V1' in df.columns , f'gru_day_V1 not in df.columns: {df.columns}'
            DB.save(df , 'pred' , 'gru_day_V1' , date)

    from src.data.update.custom.week_rank_loser import WeekRankLoserUpdater
    WeekRankLoserUpdater.update_all('recalc')

    from src.proj import PATH
    for p in PATH.result.rglob('*.tar'):
        p.unlink()
        
    for p in PATH.model.rglob('*.tar'):
        p.unlink()

        
if __name__ == '__main__':
    main()
        
    