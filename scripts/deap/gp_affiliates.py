import re
import pandas as pd
import cProfile

class Profiler(cProfile.Profile):
    def __init__(self, doso = False , builtins = True , **kwargs) -> None:
        self.doso = doso
        if self.doso: super().__init__(builtins = builtins) # type:ignore

    def __enter__(self):
        if self.doso: 
            return super().__enter__()
        else:
            return self

    def __exit__(self, *exc_info: object) -> None:
        if self.doso: return super().__exit__(*exc_info)

    def get_df(self , sort_on = 'tottime' , highlight = None , output = None):
        if not self.doso: return pd.DataFrame()
        # highlight : 'gp_math_func.py'
        df = pd.DataFrame(self.getstats(), #type:ignore
                          columns=['full_name', 'ncalls', 'ccalls', 'tottime', 'cumtime' , 'caller'])
        df.tottime = df.tottime.round(4)
        df.cumtime = df.cumtime.round(4)
        df.full_name = df.full_name.astype(str)
        df_func = pd.DataFrame([func_str_decompose(s) for s in df.full_name] , 
                                columns = ['type' , 'name' , 'where' , 'memory'])
        df = pd.concat([df_func , df],axis=1).sort_values(sort_on,ascending=False)
        df = df.loc[:,['type' , 'name' , 'ncalls', 'ccalls', 'tottime', 'cumtime' , 'where' , 'memory' , 'full_name', 'caller']]
        if isinstance(highlight , str): df = df[df.full_name.str.find(highlight) > 0]
        if isinstance(output , str): df.to_csv(output)
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
            #    print(func_string)
            break
    if data is None: 
        print(func_string)
        data = [''] * 4
    return data