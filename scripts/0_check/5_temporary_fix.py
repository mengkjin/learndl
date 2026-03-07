#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell

from src.proj.util import ScriptTool 
from src.data import DataBlock

@ScriptTool('temporary_fix')
def main(**kwargs):
    DataBlock.change_all_dumps()
        
if __name__ == '__main__':
    main()
        
    