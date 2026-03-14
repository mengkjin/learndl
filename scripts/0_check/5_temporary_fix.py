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
    from src.data.update.custom.week_rank_loser import WeekRankLoserUpdater
    WeekRankLoserUpdater.update_all('update')

    from src.proj import PATH
    for p in PATH.result.rglob('*.tar'):
        p.unlink()
        
    for p in PATH.model.rglob('*.tar'):
        p.unlink()

        
if __name__ == '__main__':
    main()
        
    