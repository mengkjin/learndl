#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Reset Trading Portfolios
# content: 更新指定组合, 输入参数为重置组合
# email: True
# mode: shell
# parameters:
#   port_name : 
#       type : Options.available_backtestports()
#       desc : backtest trade port name
#       required : True

from src.api import TradingAPI
from src.proj.util import ScriptTool

@ScriptTool('backtest_rebuild' , '@port_name')
def main(port_name : str | None = None , **kwargs):
    assert port_name is not None , 'port_name is required'
    TradingAPI.backtest_rebuild(port_name)

if __name__ == '__main__':
    main()
