import re
import pandas as pd
import cProfile

from src.proj import Logger

class EmptyTM():
    def __init__(self , obj):
        methods = [method for method in dir(obj) if callable(getattr(obj, method))]
        # 动态注册方法到新类中
        for method_name in methods:
            if method_name not in dir(self):
                method = getattr(obj, method_name)
                setattr(self, method_name, method)
    def __enter__(self):
        return self
    def __exit__(self, type, value, trace):
        if type is not None:
            Logger.error(f'Error in EmptyTM ' , type , value)
            Logger.print_exc(value)

class Profiler(cProfile.Profile):
    def __init__(self, doso = False , builtins = True , **kwargs) -> None:
        self.doso = doso
        if self.doso: 
            super().__init__(builtins = builtins) 

    def __enter__(self):
        if self.doso: 
            return super().__enter__()
        else:
            return self

    def __exit__(self, exc_type , exc_value , exc_traceback):
        if exc_type is not None:
            Logger.error(f'Error in Profiler ' , exc_type , exc_value)
            Logger.print_exc(exc_value)
        else:
            if self.doso: 
                return super().__exit__(exc_type , exc_value , exc_traceback)

    def get_df(self , sort_on = 'tottime' , highlight = None , output = None):
        if not self.doso: 
            return pd.DataFrame()
        # highlight : 'gp_math_func.py'
        df = pd.DataFrame(
            getattr(self , 'getstats')(), 
            columns=['full_name', 'ncalls', 'ccalls', 'cumtime' , 'tottime' , 'caller']).astype({'full_name':str})
        df['tottime'] = df['tottime'].round(4)
        df['cumtime'] = df['cumtime'].round(4)
        df_func = pd.DataFrame(
            [func_str_decompose(s) for s in df.full_name] , 
            columns = pd.Index(['type' , 'name' , 'where' , 'memory']))
        df = pd.concat([df_func , df],axis=1).sort_values(sort_on,ascending=False)
        column_order = ['type' , 'name' , 'ncalls', 'ccalls', 'cumtime' , 'tottime' , 'where' , 'memory' , 'full_name', 'caller']
        df = df.loc[:,column_order]
        if isinstance(highlight , str): 
            df = df[df.full_name.str.find(highlight) > 0]
        if isinstance(output , str): 
            df.to_csv(output)
        return df

def func_str_decompose(func_string):
    # Define the regular expression pattern to extract information
    pattern = {
        r'<code object (.+) at (.+), file (.+), line (\d+)>' : ['function' , (0,) , (2,3) , (1,)] ,
        r'<function (.+) at (.+)>' : ['function' , (0,) , () , ()] ,
        r'<method (.+) of (.+) objects>' : ['method' , (0,) , (1,) , ()] ,
        r'<built-in method (.+)>' : ['built-in-method' , (0,) , () , ()] ,
        r'<fastparquet.(.+)>' : ['fastparquet' , () , () , ()] ,
        r'<pandas._libs.(.+)>' : ['pandas._libs' , (0,) , () , ()],
    }
    data = None
    for pat , use in pattern.items():
        match = re.match(pat, func_string)
        if match:
            data = [use[0] , ','.join(match.group(i+1) for i in use[1]) , 
                    ','.join(match.group(i+1) for i in use[2]) , ','.join(match.group(i+1) for i in use[3])]
            #try:
            #    data = [use[0] , ','.join(match.group(i+1) for i in use[1]) , ','.join(match.group(i+1) for i in use[2])]
            #except:
            #    Logger.stdout(func_string)
            break
    if data is None: 
        Logger.stdout(func_string)
        data = [''] * 4
    return data