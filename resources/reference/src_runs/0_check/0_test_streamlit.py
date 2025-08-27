#! /usr/bin/env python3.10
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


import sys , pathlib , time , random
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.basic import AutoRunTask
from src_app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder(txt = 'Bye, World!' , email = 0)
@ScriptLock('test_streamlit' , timeout = 10)
def main(txt : str = 'Hello, World!' , **kwargs):
    with AutoRunTask(f'test streamlit' , **kwargs) as runner:
        runner.info(str(kwargs))
        runner.info(f'info:{txt}')
        runner.warning(f'warning:{txt}')
        runner.debug(f'debug:{txt}')
        runner.critical(f'critical:{txt}')
        if (rnd := random.random()) < 0.5:
            runner.error(f'error:{rnd}')
        else:
            runner.info(f'info:{rnd}')
        time.sleep(5)

    return runner
        
if __name__ == '__main__':
    main()
        
    