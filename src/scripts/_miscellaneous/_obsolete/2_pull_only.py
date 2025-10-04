#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: pull only
# content: 自动拉取最新代码
# email: False
# mode: shell

import subprocess 
from src.app import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    subprocess.run("git pull", shell=True, check=True)

    return 'Finish pull'

if __name__ == '__main__':
    main()