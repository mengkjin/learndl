#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: pull only
# content: 放弃当前所有更改并自动拉取最新代码
# email: False
# mode: shell

import subprocess , shutil
from pathlib import Path
from src.proj import Logger
from src.interactive.backend import BackendTaskRecorder

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
    for folder in [*Path('src').rglob('*/') , *Path('configs').rglob('*/')][::-1]:
        if folder.is_dir() and not [x for x in folder.iterdir() if x.name != '__pycache__']:
            subfiles = [x for x in folder.rglob('*') if x.is_file()]
            if not len(subfiles):
                Logger.stdout(f"Removing empty folder: {folder}")
                folder.rmdir()
            else:
                if all([x.suffix == '.pyc' for x in subfiles]):
                    Logger.stdout(f"Removing folder with only pyc files: {folder}")
                    shutil.rmtree(folder)
                else:
                    Logger.error(f"Error removing folder: {folder}:")
                    Logger.error(f"Subfiles: {subfiles}")

    return f'Finish pull: {result.stdout}'

if __name__ == '__main__':
    main()