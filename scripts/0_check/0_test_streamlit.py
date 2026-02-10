#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: False
# mode: shell

from src.proj import Logger
from src.proj.util import ScriptTool 

@ScriptTool('test_streamlit' , '@port_name' , txt = 'Bye, World!' , lock_num = 2 , lock_timeout = 10)
def main(port_name : str = 'a' , **kwargs):
    Logger.test_logger()
        
if __name__ == '__main__':
    main()
        
    