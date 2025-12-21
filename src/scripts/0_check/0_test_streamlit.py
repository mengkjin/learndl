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
from src.proj import Logger , LogWriter , Timer
from src.app import ScriptTool

@ScriptTool('test_streamlit' , '@port_name' , txt = 'Bye, World!' , lock_num = 2 , lock_timeout = 10)
def main(port_name : str = 'a' , module_name : str = 'bbb' , txt : str = 'Hello, World!' , start : int | None = 100 , **kwargs):
    catcher = LogWriter('log.txt')
    with catcher:
        Logger.stdout('This will be caught')
        with Logger.ParagraphIII('main part'):
            with Timer('abc'):
                Logger.critical('critical message')
                Logger.error('error message')
                Logger.warning('warning message')
                Logger.info('info message')
                Logger.debug('debug message')
                Logger.highlight('highlight message')
                Logger.highlight('highlight message with default prefix' , default_prefix = True)
                Logger.divider()
                Logger.success('success message')
                Logger.failure('failure message')
                Logger.marking('marking message')
                Logger.attention('attention message')
                Logger.conclude(f'cmd is: {" ".join(sys.argv)}' , level = 'info')
                Logger.conclude(f'this is kwargs: {str(kwargs)}' , level = 'info')
                Logger.conclude(f'this is an info: {txt}' , level = 'info')
                Logger.conclude(f'critical: lazy message')
                Logger.conclude(f'this is a warning: {txt}' , level = 'warning')
                Logger.conclude(f'this is a critical: {txt}' , level = 'critical')
                Logger.stdout(f'email: {ScriptTool.get_value('email')}')
                Logger.stdout(f'forfeit_task: {ScriptTool.get_value('forfeit_task')}')
                Logger.stdout(f'task_key: {ScriptTool.get_value('task_key')}')
                if (rnd := random.random()) < 0.5:
                    Logger.conclude(f'this is an error: random number {rnd} < 0.5' , level = 'error')
                else:
                    Logger.conclude(f'this is an info: random number {rnd} >= 0.5' , level = 'info')
    time.sleep(1)
        
if __name__ == '__main__':
    main()
        
    