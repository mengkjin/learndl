import polars as pl

from typing import Literal

from src.proj import DB 
from src.proj.func import singleton
from src.data.util import INFO

from .access import DateDataAccess

@singleton
class MinKLineAccess(DateDataAccess):
    MAX_LEN = 60
    PL_DATA_TYPE_LIST = ['min' , '5min']
    
    def data_loader(self , date , data_type):
        if data_type in self.PL_DATA_TYPE_LIST: 
            df = DB.load('trade_ts' , data_type , date , vb_level = 99 , use_alt = True)
            if not df.empty: 
                secid = INFO.get_secid(date) # noqa
                df = df.query('secid in @secid')
        else:
            raise KeyError(data_type)
        return df
    
    @staticmethod
    def reform_mkline(df : pl.DataFrame , with_ret : bool = False):
        df = df.with_columns(pl.col('secid').cast(pl.Int64))
        if with_ret:
            df = df.with_columns(
                pl.when(pl.col('minute') == 0).then(pl.col('open')).\
                    otherwise(pl.col('close').shift(1)).over('secid').alias('preclose')
            ).with_columns(
                ((pl.col('close') - pl.col('preclose')) / pl.col('preclose')).alias('ret')
            )
        return df

    def get_1min(self , date , with_ret = False):
        df = self.get_pl(date , 'min')
        df = self.reform_mkline(df , with_ret)
        return df
    
    def get_5min(self , date , with_ret = False):
        df = self.get_pl(date , '5min')
        df = self.reform_mkline(df , with_ret)
        return df
    
    def get_kline(self , date , with_ret = False):
        df = self.get_1min(date , with_ret)
        if df.shape[0] == 0: 
            df = self.get_5min(date , with_ret)
        return df

    def get_inday_corr(self , date : int , 
                       val1 : Literal['ret' , 'volume' , 'mkt' , 'vwap'] | str, 
                       val2 : Literal['ret' , 'volume' , 'mkt' , 'vwap'] | str,
                       lag1 : int = 0 , lag2 : int = 0 ,
                       rename : str = 'value' , beta = False):  
        
        assert val1 != val2 or lag1 != lag2 , \
            f'val1 and val2 must be different or lag1 and lag2 must be different , got {val1} , {val2} , {lag1} , {lag2}'

        df = self.get_kline(date , with_ret = (val1 in ['ret' , 'mkt']) or (val2 in ['ret' , 'mkt']))
        if val1 == 'mkt' or val2 == 'mkt':
            mkt = df.group_by('minute').agg(pl.col('ret').mean().alias('mkt'))
            df = df.join(mkt, on='minute')

        df = df.with_columns(
            pl.col(val1).shift(lag1).over('secid').alias('val1') ,
            pl.col(val2).shift(lag2).over('secid').alias('val2') ,
        )
        if beta:
            df = df.group_by('secid').agg([(pl.corr('val1', 'val2') * pl.std('val1') / pl.std('val2')).alias(rename)])
        else:
            df = df.group_by('secid').agg([pl.corr('val1', 'val2').alias(rename)])
        return df
        
MKLINE = MinKLineAccess()