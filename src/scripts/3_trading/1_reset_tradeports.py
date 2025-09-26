#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Reset Trading Portfolios
# content: 更新指定组合, 输入参数为重置组合
# email: True
# mode: shell
# parameters:
#   reset_port_name : 
#       type : "[p.name for p in Path('data/Export/trading_portfolio').iterdir() if not p.name.startswith('.')]"
#       prefix : "tradeport/"
#       desc : port names by ","
#       required : True

from src.res.api import TradingAPI
from src.app.script_tool import ScriptTool

@ScriptTool('reset_tradeports' , '@reset_port_name')
def main(reset_port_name : str , **kwargs):
    TradingAPI.update([reset_port_name])

if __name__ == '__main__':
    main()
