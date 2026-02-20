#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from data.update.custom.daily_risk import DailyRiskUpdater
from data.update.custom.market_daily_risk import MarketDailyRiskUpdater
from src.proj.util import ScriptTool 

@ScriptTool('temporary_fix')
def main(**kwargs):
    DailyRiskUpdater.update_all('recalc')
    MarketDailyRiskUpdater.update_all('recalc')
        
if __name__ == '__main__':
    main()
        
    