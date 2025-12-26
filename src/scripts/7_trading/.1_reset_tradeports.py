#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Reset Trading Portfolios
# content: 更新指定组合, 输入参数为重置组合
# email: True
# mode: shell
# parameters:
#   reset_port_name : 
#       type : Options.available_tradeports()
#       prefix : "tradeport/"
#       desc : port names by ","
#       required : True

from src.api import TradingAPI
from src.basic import ScriptTool

@ScriptTool('reset_tradeports' , '@reset_port_name')
def main(reset_port_name : str | None = None , **kwargs):
    assert reset_port_name is not None , 'reset_port_name is required'
    TradingAPI.update([reset_port_name])

if __name__ == '__main__':
    main()
