#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: pull only
# content: 放弃当前所有更改并自动拉取最新代码
# email: False
# mode: shell

import subprocess 
from src.app import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    print("🧹 Clean local changes...")
    subprocess.run(['git', 'reset', '--hard', 'HEAD'], check=True)
    subprocess.run(['git', 'clean', '-fd'], check=True)
    
    print("⬇️ Pull latest code...")
    result = subprocess.run(['git', 'pull'], capture_output=True, text=True, check=True)
    print(f"✅ Done: {result.stdout}")

    return f'Finish pull: {result.stdout}'

if __name__ == '__main__':
    main()