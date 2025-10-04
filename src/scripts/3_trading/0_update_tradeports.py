#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Update Trading Portfolios
# content: 更新所有交易组合, 无重置指定组合
# email: True
# mode: shell

from src.res.api import TradingAPI
from src.app.script_tool import ScriptTool

@ScriptTool('update_tradeports')
def main(**kwargs):
    TradingAPI.update()
    
if __name__ == '__main__':
    main()
