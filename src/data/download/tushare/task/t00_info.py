# do not use relative import in this file because it will be running in top-level directory
import numpy as np
import pandas as pd

from typing import Any

from src.data.download.tushare.basic import InfoFetcher , ts_code_to_secid
 
class Calendar(InfoFetcher):
    """A share calendar"""
    DB_KEY = 'calendar'

    def get_data(self , date = None):
        renamer = {'cal_date' : 'calendar' ,
                   'is_open'  : 'trade'}
        fields : str | Any = list(renamer.keys()) if renamer else None
        df = self.pro.trade_cal(fields = fields , exchange='SSE').rename(columns=renamer)
        df = df.astype({'calendar':int}).sort_values('calendar').reset_index(drop = True)

        # process
        trd = df.query('trade == 1').reset_index(drop=True)
        trd['pre'] = trd['calendar'].shift(1, fill_value=-1)
        trd = df.merge(trd.drop(columns='trade') , on = 'calendar' , how = 'left').astype(int).ffill()
        trd['td_index'] = trd['trade'].cumsum()

        return trd
    
class Description(InfoFetcher):
    """A share description"""
    DB_KEY = 'description'
    def get_data(self , date = None):
        renamer = {
            'ts_code' : 'ts_code' ,
            'name':'sec_name' ,
            'exchange':	'exchange_name'	,
            'list_date' : 'list_dt' ,
            'delist_date' : 'delist_dt' ,
            'industry' : 'industry' ,
        }
        fields : str | Any = list(renamer.keys())
        df = pd.concat([
            self.pro.stock_basic(fields = fields , list_status = 'L') ,
            self.pro.stock_basic(fields = fields , list_status = 'D') ,
            self.pro.stock_basic(fields = fields , list_status = 'P')
        ]).rename(columns=renamer)

        df = ts_code_to_secid(df , drop_old=False).fillna({'list_dt':-1 , 'delist_dt':99991231}).astype({'list_dt':int , 'delist_dt':int}).reset_index(drop=True)
        return df
    
class SWIndustry(InfoFetcher):
    """A share sw industry"""
    DB_KEY = 'industry' 
    def get_data(self , date = None):

        df1 = self.iterate_fetch(self.pro.index_member_all , limit = 2000 , is_new = 'Y')
        df2 = self.iterate_fetch(self.pro.index_member_all , limit = 2000 , is_new = 'N')

        df = pd.concat([df1 , df2])
        df = ts_code_to_secid(df).fillna({'in_date':99991231 , 'out_date':99991231}).astype({'in_date':int , 'out_date':int}).reset_index(drop=True)
        return df

class ZXIndustry(InfoFetcher):
    """A share zx industry"""
    DB_KEY = 'zx_industry' 
    def get_data(self , date = None):

        df1 = self.iterate_fetch(self.pro.ci_index_member , limit = 2000 , is_new = 'Y')
        df2 = self.iterate_fetch(self.pro.ci_index_member , limit = 2000 , is_new = 'N')

        df = pd.concat([df1 , df2])
        df = ts_code_to_secid(df).fillna({'in_date':99991231 , 'out_date':99991231}).astype({'in_date':int , 'out_date':int}).reset_index(drop=True)
        return df
    
class ChangeName(InfoFetcher):
    """A share change name table (for name change and get st/*st)"""
    DB_KEY = 'change_name'      
    def get_data(self , date = None):

        df = self.iterate_fetch(self.pro.namechange , limit = 5000)
        df = ts_code_to_secid(df).fillna({'start_date':-1 , 'ann_date':-1 , 'end_date':99991231}).astype({'start_date':int , 'ann_date':int , 'end_date':int})
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
        return ['撤销ST', '撤销*ST', '摘星', '摘星改名', '恢复上市加N', '恢复上市']