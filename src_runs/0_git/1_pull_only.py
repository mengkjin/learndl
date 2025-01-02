#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: pull only
# content: 自动拉取最新代码
# email: False
# close_after_run: False

import subprocess 
if __name__ == '__main__':
    subprocess.run("git pull", shell=True, check=True)