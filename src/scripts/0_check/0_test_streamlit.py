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
import time , random , sys

from src.proj import Logger , LogWriter
from src.basic import Timer
from src.app.script_tool import ScriptTool

@ScriptTool('test_streamlit' , '@port_name' , txt = 'Bye, World!' , lock_num = 2 , lock_timeout = 10)
def main(port_name : str = 'a' , module_name : str = 'bbb' , txt : str = 'Hello, World!' , start : int | None = 100 , **kwargs):
    catcher = LogWriter('log.txt')
    with catcher:
        print('This will be caught')
        with Logger.EnclosedMessage('main part'):
            with Timer('abc'):
                ScriptTool.info(f'cmd is: {" ".join(sys.argv)}')
                ScriptTool.info(f'this is kwargs: {str(kwargs)}')
                ScriptTool.info(f'this is an info: {txt}')
                Logger.cache_message('critical' , f'critical: lazy message')
                ScriptTool.warning(f'this is a warning: {txt}')
                ScriptTool.debug(f'this is a debug: {txt}')
                ScriptTool.critical(f'this is a critical: {txt}')
                print(f'email: {ScriptTool.get_value('email')}')
                print(f'forfeit_task: {ScriptTool.get_value('forfeit_task')}')
                print(f'task_key: {ScriptTool.get_value('task_key')}')
                if (rnd := random.random()) < 0.5:
                    ScriptTool.error(f'this is an error: {rnd}')
                else:
                    ScriptTool.info(f'this is an info: {rnd}')
    time.sleep(2)
        
if __name__ == '__main__':
    main()
        
    