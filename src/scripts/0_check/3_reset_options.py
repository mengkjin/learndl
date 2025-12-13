#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: pull only
# content: 放弃当前所有更改并自动拉取最新代码
# email: False
# mode: shell

from src.app import BackendTaskRecorder
from src.proj import Options

@BackendTaskRecorder()
def main(**kwargs):
    Options.cache.clear()
    print(f'Success : reset options cache')

if __name__ == '__main__':
    main()