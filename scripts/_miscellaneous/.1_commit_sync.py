#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: commit and sync
# content: 自动提交所有修改,并同步(rebase+push)
# email: False
# mode: shell
# parameters:
#   additional_message : 
#       type : str
#       desc : additional message

import subprocess ,  socket
from datetime import datetime
from src.interactive.backend import BackendTaskRecorder

def run_command(command: str):
    subprocess.run(command, shell=True, check=True)

@BackendTaskRecorder()
def main(additional_message : str | list[str] = '' , **kwargs):
    prefixes = [socket.gethostname() , datetime.now().strftime('%Y%m%d')]
    if isinstance(additional_message , str): 
        additional_message = [additional_message]
    commit_message = ','.join([msg for msg in prefixes + additional_message if msg])

    run_command("git add .")
    run_command(f"git commit -m '{commit_message}'")
    run_command("git pull --rebase")
    run_command("git push")

    return f'Finish commit and sync: {commit_message}'

if __name__ == '__main__':
    main()