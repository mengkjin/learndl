#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# mode: shell
# parameters:
#   port_name : 
#       type : [a , b , c]
#       desc : trade port name
#       default : a
#   module_name : 
#       type : str
#       desc : module to train
#       default : bbb
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
#   forget :
#       type : bool
#       desc : forget
#       required : False
#       default : None
#   start : 
#       type : int
#       desc : yyyymmdd (or -1)
#       min : -1
#   end : 
#       type : int
#       desc : yyyymmdd (or -1)
#       max : 99991231
#   seed : 
#       type : float
#       desc : seed
#       required : False
#       default : 42.
from src.proj import Logger
from src.proj.util import ScriptTool 

@ScriptTool('test_streamlit' , '@port_name' , txt = 'Bye, World!' , lock_num = 2 , lock_timeout = 10)
def main(port_name : str = 'a' , module_name : str = 'bbb' , txt : str = 'Hello, World!' , start : int | None = 100 , **kwargs):
    Logger.test_logger()
        
if __name__ == '__main__':
    main()
        
    