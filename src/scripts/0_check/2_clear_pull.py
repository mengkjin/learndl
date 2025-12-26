#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: pull only
# content: 放弃当前所有更改并自动拉取最新代码
# email: False
# mode: shell

import subprocess
from pathlib import Path
from src.proj import Logger
from src.app import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    # clean local changes
    Logger.highlight("Clean local changes...")
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], check=True)
    subprocess.run(['git', 'clean', '-fd'], check=True)
    
    # pull latest code
    Logger.highlight("Pull latest code...")
    result = subprocess.run(['git', 'pull'], capture_output=True, text=True, check=True)
    Logger.success(f"Done: {result.stdout}")

    # remove empty folders in src
    Logger.highlight("Remove empty script folders in src...")
    for folder in Path('src').rglob('*/'):
        if folder.is_dir() and not list(folder.iterdir()):
            Logger.success(f"Remove empty folder: {folder}")
            folder.rmdir()

    return f'Finish pull: {result.stdout}'

if __name__ == '__main__':
    main()