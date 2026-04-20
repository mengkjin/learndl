#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-04-20
# description: Rebuild all backtest portfolios
# content: 重建所有回测组合
# email: True
# mode: shell
# parameters:
#   confirm : 
#       type : bool
#       desc : confirm to rebuild all backtest portfolios
#       required : True
#       default : False

from src.api import TradingAPI
from src.proj.util import ScriptTool

@ScriptTool('backtest_rebuild_all')
def main(confirm : bool | None = None , **kwargs):
    assert confirm , 'confirm is required'
    TradingAPI.backtest_rebuild_all()

if __name__ == '__main__':
    main()
