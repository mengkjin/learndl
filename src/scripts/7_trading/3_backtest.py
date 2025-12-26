#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-11-30
# description: Backtest Trading Portfolios
# content: 回测交易组合
# email: True
# mode: shell
# parameters:
#   port_name_starter : 
#       type : str
#       desc : trade port name (search for available tradeport names which starts with this)
#       required : True
#   start : 
#       type : int
#       desc : start yyyymmdd (or -1)
#   end : 
#       type : int
#       desc : end yyyymmdd

from src.api import TradingAPI
from src.basic import ScriptTool

@ScriptTool('analyze_tradeport' , '@port_name_starter' , lock_num = 0)
def main(port_name_starter : str | None = None , start : int | None = None , end : int | None = None , **kwargs):
    assert port_name_starter is not None , 'port_name_starter is required'
    TradingAPI.Backtest(port_name_starter, start , end)
    
if __name__ == '__main__':
    main()
