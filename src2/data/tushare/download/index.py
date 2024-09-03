import pandas as pd

from ..basic import pro , code_to_secid , MonthFetecher

class THSConcept(MonthFetecher):
    '''Tonghuashun Concept'''
    def db_src(self): return 'membership_ts'
    def db_key(self): return 'concept'    
    def get_data(self , date : int):
        df_theme = pd.concat([pro.ths_index(exchange = 'A', type = 'N') , 
                              pro.ths_index(exchange = 'A', type = 'TH')]).reset_index(drop=True)
        dfs = []
        for i , ts_code in enumerate(df_theme['ts_code']):
            # print(i , ts_code)
            df = pro.ths_member(ts_code = ts_code)
            dfs.append(df)
        df_all = pd.concat(dfs).rename(columns={'name':'concept'})
        df_all = df_all.merge(df_theme , on = 'ts_code' , how='left').rename(columns={'ts_code':'index_code'})
        df_all = code_to_secid(df_all , 'code')
        df = df_all.reset_index(drop = True)
        return df
    
    