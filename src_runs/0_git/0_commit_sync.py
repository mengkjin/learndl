#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: commit and sync
# content: 自动提交所有修改,并同步(rebase+push)
# email: False
# close_after_run: False
# param_inputs:
#   additional_message : 
#       type : str
#       desc : additional message

import sys 

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

import subprocess ,  socket
from datetime import datetime
from src_runs.widget import argparse_dict

if __name__ == '__main__':
    params = argparse_dict()
    additional_message = params.get('additional_message' , '')
    prefixes = [socket.gethostname() , datetime.now().strftime('%Y%m%d')]
    if isinstance(additional_message , str): additional_message = [additional_message]
    commit_message = ','.join([msg for msg in prefixes + additional_message if msg])

    subprocess.run("git add .", shell=True, check=True)
    subprocess.run(f"git commit -m '{commit_message}'", shell=True, check=True)
    subprocess.run("git pull --rebase", shell=True, check=True)
    subprocess.run("git push", shell=True, check=True)
