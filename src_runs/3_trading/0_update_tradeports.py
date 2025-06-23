#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Update Trading Portfolios
# content: 更新所有交易组合, 无重置指定组合
# email: True
# close_after_run: False

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import TradingAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    with AutoRunTask('update trading portfolios' , **params , email_if_attachment = True , message_capturer=True) as runner:
        TradingAPI.update()

if __name__ == '__main__':
    main()
