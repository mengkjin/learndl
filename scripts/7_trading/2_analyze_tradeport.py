#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-06-22
# description: Analyze Trading Portfolios
# content: 分析交易组合
# email: False
# mode: shell
# parameters:
#   port_name : 
#       type : Options.available_tradeports()
#       prefix : "tradeport/"
#       desc : trade port name
#       required : True
#   start : 
#       type : int
#       desc : start yyyymmdd (or -1)
#   end : 
#       type : int
#       desc : end yyyymmdd

from src.api import TradingAPI
from src.proj.util import ScriptTool

@ScriptTool('analyze_tradeport' , '@port_name' , lock_num = 0)
def main(port_name : str | None = None , start : int | None = None , end : int | None = None , **kwargs):
    assert port_name is not None , 'port_name is required'
    TradingAPI.Analyze(port_name, start , end)
    
if __name__ == '__main__':
    main()
