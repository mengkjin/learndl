#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-06-22
# description: Analyze Trading Portfolios
# content: 分析交易组合
# email: False
# mode: shell
# parameters:
#   port_name : 
#       type : "[p.name for p in Path('data/Export/trading_portfolio').iterdir() if not p.name.startswith('.')]"
#       prefix : "tradeport/"
#       desc : trade port name
#       required : True
#   start : 
#       type : int
#       desc : start yyyymmdd (or -1)
#   end : 
#       type : int
#       desc : end yyyymmdd

from src.res.api import TradingAPI
from src.app.script_tool import ScriptTool

@ScriptTool('analyze_tradeport' , '@port_name' , lock_num = 0)
def main(port_name : str , start : int | None = None , end : int | None = None , **kwargs):
    TradingAPI.Analyze(port_name, start , end)
    
if __name__ == '__main__':
    main()
