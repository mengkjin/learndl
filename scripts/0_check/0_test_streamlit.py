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
import random , sys , tqdm
from src.proj import Logger
from src.proj.util import ScriptTool

import pandas as pd
import matplotlib.pyplot as plt

@ScriptTool('test_streamlit' , '@port_name' , txt = 'Bye, World!' , lock_num = 2 , lock_timeout = 10)
def main(port_name : str = 'a' , module_name : str = 'bbb' , txt : str = 'Hello, World!' , start : int | None = 100 , **kwargs):
    Logger.stdout('This will be caught')
    with Logger.ParagraphIII('main part'):
        with Logger.Timer('abc'):
            print('this is a print message')
            Logger.critical('critical message')
            Logger.error('error message')
            Logger.warning('warning message')
            Logger.info('info message')
            Logger.debug('debug message')
            Logger.highlight('highlight message')
            Logger.highlight('highlight message with default prefix' , prefix = True)
            Logger.divider()
            Logger.success('success message' , vb_level = 5)
            Logger.remark('remark message' , vb_level = 12)
            Logger.remark('remark message with default prefix' , color = 'lightblue')
            Logger.footnote('footnote message' , vb_level = 99)
            Logger.alert1('warning message')
            Logger.alert2('error message')
            Logger.alert3('critical message')
            Logger.skipping('skipping message' , indent = 1)
            Logger.conclude(f'cmd is: {" ".join(sys.argv)}' , level = 'info')
            Logger.conclude(f'this is kwargs: {str(kwargs)}' , level = 'info')
            Logger.conclude(f'this is an info: {txt}' , level = 'info')
            Logger.conclude(f'this is an debug: {txt}' , level = 'debug')
            Logger.conclude(f'critical: lazy message')
            Logger.conclude(f'this is a warning: {txt}' , level = 'warning')
            Logger.conclude(f'this is a error: {txt}' , level = 'error')
            Logger.conclude(f'this is a critical: {txt}' , level = 'critical')
            Logger.stdout(f'email: {ScriptTool.get_value('email')}' , indent = 0)
            Logger.stdout(f'forfeit_task: {ScriptTool.get_value('forfeit_task')}' , indent = 1)
            Logger.stdout(f'task_key: {ScriptTool.get_value('task_key')}' , indent = 2)
            Logger.log_only('this is a log only message')

            df = pd.DataFrame({'a':[1,2,3,4,5,6,7,8,9,10],'b':[4,5,6,7,8,9,10,11,12,13]})
            Logger.Display(df)
            fig = plt.figure()
            plt.plot([1,2,3],[4,5,6])
            plt.close(fig)
            Logger.Display(fig)

            for i in tqdm.tqdm(range(100) , desc='processing'):
                pass

            if (rnd := random.random()) < 0.5:
                Logger.conclude(f'this is an error: random number {rnd} < 0.5' , level = 'error')
            else:
                Logger.conclude(f'this is an info: random number {rnd} >= 0.5' , level = 'info')

            raise Exception('test exception')
        
if __name__ == '__main__':
    main()
        
    