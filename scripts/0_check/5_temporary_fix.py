#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 
from src.proj import PATH
import shutil

@ScriptTool('temporary_fix')
def main(**kwargs):
    for path in PATH.model.rglob('detailed_alpha'):
        if path.is_dir():
            shutil.rmtree(path)

    for path in PATH.result.rglob('*.tar'):
        path.unlink()
        
if __name__ == '__main__':
    main()
        
    