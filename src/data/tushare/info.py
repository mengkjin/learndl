import numpy as np
import pandas as pd

from typing import Any , Literal

from .basic import pro , code_to_secid , InfoFetecher
 
class Calendar(InfoFetecher):
    def db_src(self): return 'information_ts'
    def db_key(self): return 'calendar'
    def get_data(self , date):
        renamer = {'cal_date' : 'calendar' ,
                   'is_open'  : 'trade'}
        fields : str | Any = list(renamer.keys()) if renamer else None
        df = pro.trade_cal(fields = fields , exchange='SSE').rename(columns=renamer)
        df = df.sort_values('calendar').reset_index(drop = True)
        return df
    
class Description(InfoFetecher):
    def db_src(self): return 'information_ts'
    def db_key(self): return 'description'
    def get_data(self):
        renamer = {
            'ts_code' : 'ts_code' ,
            'name':'sec_name' ,
            'exchange':	'exchange_name'	,
            'list_date' : 'list_dt' ,
            'delist_date' : 'delist_dt'
        }
        fields : str | Any = list(renamer.keys())
        df = pd.concat([
            pro.stock_basic(fields = fields , list_status = 'L') ,
            pro.stock_basic(fields = fields , list_status = 'D') ,
            pro.stock_basic(fields = fields , list_status = 'P')
        ]).rename(columns=renamer)

        df = code_to_secid(df , retain=True)
        df['list_dt'] = df['list_dt'].fillna(-1).astype(int)
        df['delist_dt'] = df['delist_dt'].fillna(99991231).astype(int)
        df = df.reset_index(drop = True)
        return df
    
class SWIndustry(InfoFetecher):
    def db_src(self): return 'information_ts'
    def db_key(self): return 'industry'    
    def get_data(self):
        limit = 2000
        args_is_new = ['Y' , 'N']
        dfs = []
        for is_new in args_is_new:
            offset = 0
            while True:
                df = pro.index_member_all(is_new = is_new , limit = limit , offset = offset)
                if len(df) == 0: break
                dfs.append(df)
                offset += limit

        df = pd.concat(dfs)
        df = code_to_secid(df)
        df['in_date'] = df['in_date'].fillna(99991231).astype(int)
        df['out_date'] = df['out_date'].fillna(99991231).astype(int)
        df = df.reset_index(drop=True)
        return df
    
class ChangeName(InfoFetecher):
    def db_src(self): return 'information_ts'
    def db_key(self): return 'industry'    
    def get_data(self):
        limit = 5000
        dfs = []
        offset = 0
        while True:
            df = pro.namechange(limit = limit , offset = offset)
            if len(df) == 0: break
            dfs.append(df)
            offset += limit
        df = pd.concat(dfs).reset_index(drop = True)
        df = code_to_secid(df)
        df['start_date'] = df['start_date'].fillna(-1).astype(int)
        df['ann_date'] = df['ann_date'].fillna(-1).astype(int)
        df['end_date'] = df['end_date'].fillna(99991231).astype(int)
        df['entry_dt'] = np.where(df['ann_date'] > 0 , np.minimum(df['start_date'] , df['ann_date']) , df['ann_date'])
        df['remove_dt'] = df['end_date']
        assert df['change_reason'].isin(self._dangerous_type() + self._safe_type()).all , \
            df['change_reason'][~df['change_reason'].isin(self._dangerous_type() + self._safe_type())]
        return df
    
    @staticmethod
    def _dangerous_type():
        return ['终止上市', 'ST', '*ST', '暂停上市']

    @staticmethod
    def _safe_type():
        return ['撤销ST', '其他', '撤销*ST', '摘星', '改名', '摘星改名',
                '恢复上市加N', '恢复上市', '更名', '完成股改', '摘G', '未股改加S']