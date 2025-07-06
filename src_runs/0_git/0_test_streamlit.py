#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: test me
# content: very short script to test streamlit
# email: True
# close_after_run: True
# param_inputs:
#   port_name : 
#       type : [a , b , c]
#       desc : trade port name
#       required : True
#       default : a
#   module_name : 
#       type : str
#       desc : module to train
#       required : True
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


import sys , pathlib , time
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.basic.util import Logger
from src_runs.util import argparse_dict , ExitMessenger

def main():
    params = argparse_dict()
    time.sleep(1)  # small delay to ensure status update
                
    Logger.info("Hello, World!")
    pathlib.Path('test.txt').touch()
    ExitMessenger.update(params.get('task_id') , files = ['test.txt'], code = 0, message = 'Hello, World!')
        
if __name__ == '__main__':
    main()
        
    