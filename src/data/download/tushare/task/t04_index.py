# do not use relative import in this file because it will be running in top-level directory
import pandas as pd
import numpy as np

from src.data.download.tushare.basic import InfoFetcher , DayFetcher ,MonthFetcher , RollingFetcher , TimeSeriesFetcher , ts_code_to_secid
from src.basic import DB , CALENDAR
from src.proj import PATH
from typing import Any


def index_weight_get_data(instance : RollingFetcher , index_code , start_dt , end_dt , limit = 4000):
    """get index weight data by iterate fetch"""
    assert start_dt is not None and end_dt is not None , 'start_dt and end_dt must be provided'
    df = instance.iterate_fetch(instance.pro.index_weight , limit = limit , max_fetch_times = 500 , index_code=index_code , 
                            start_date = str(start_dt) , end_date = str(end_dt))
    if df.empty: 
        return df
    df = ts_code_to_secid(df , 'con_code').rename(columns={'trade_date':instance.ROLLING_DATE_COL})
    df = df.sort_values([instance.ROLLING_DATE_COL ,'weight'] , ascending=False)
    df['weight'] = df.groupby(instance.ROLLING_DATE_COL)['weight'].transform(lambda x: x / x.sum())
    return df

class IndexBasic(InfoFetcher):
    """index basic infomation"""
    DB_KEY = 'index_basic'
    UPDATE_FREQ = 'w'

    def get_data(self , date):
        markets = ['MSCI-MSCI指数' , 'CSI-中证指数' , 'SSE-上交所指数' , 'SZSE-深交所指数' , 'CICC-中金指数' , 'SW-申万指数' , 'OTH-其他指数']
        
        dfs : list[pd.DataFrame] = []
        for market in markets:
            df = self.iterate_fetch(self.pro.index_basic , limit = 5000 , exchange=market.split('-')[0])
            dfs.append(df)

        df = pd.concat([d for d in dfs if not d.empty]).drop_duplicates()
        return df

class IndexDaily(TimeSeriesFetcher):
    """
    index daily quotes
    000300.SH: 沪深300指数
    000905.SH: 中证500指数
    000906.SH: 中证800指数
    000852.SH: 中证1000指数
    932000.CSI: 中证2000指数
    000985.CSI: 中证全指指数
    """
    START_DATE = 20100101
    DB_SRC = 'index_daily_ts'
    DB_KEY = 'ignore'
    TARGET_INDEX = ['000300.SH' , '000905.SH' , '000906.SH' , '000852.SH' , '932000.CSI' , '000985.CSI']

    def last_update_date(self) -> int:
        target_path = DB.path(self.DB_SRC , self.TARGET_INDEX[-1])
        ldate = PATH.file_modified_date(target_path , self.START_DATE)
        if self.rollback_date: 
            ldate = min(ldate , self.rollback_date)
        return ldate
    
    def get_data(self , index : str , start_date : int , end_date : int):
        if start_date > end_date:
            return pd.DataFrame()
        if self.rollback_date is not None:
            start_date = min(start_date , self.rollback_date)
        df = self.iterate_fetch(self.pro.index_daily , limit = 5000 , ts_code = index , start_date = str(start_date) , end_date = str(end_date))
        return df

    def target_dates(self):
        """get update dates for rolling fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        update_to = CALENDAR.update_to()
        update = self.updatable(self.last_update_date() , self.UPDATE_FREQ , update_to)
        return [update_to] if update else []

    def update_dates(self , dates):
        """override TushareFetcher.update_with_dates because rolling fetcher needs get data by ROLLING_SEP_DAYS intervals"""
        if not dates:
            return
        end_dt = max(dates)

        for index in self.TARGET_INDEX:
            path = DB.path(self.DB_SRC , index)
            old_df = pd.DataFrame()
            start_date = self.START_DATE
            if path.exists():
                _df = DB.load_df(path)
                if 'trade_date' in old_df.columns:
                    start_date = int(old_df['trade_date'].max()) + 1
                    old_df = _df
            df = self.get_data(index , start_date , end_dt)
            df = pd.concat([old_df , df]).drop_duplicates(subset = ['trade_date']).sort_values('trade_date')
            df['trade_date'] = df['trade_date'].astype(int)
            DB.save(df , self.DB_SRC , index , verbose = True)

class ZXIndexDaily(DayFetcher):
    """
    index daily quotes
    """
    START_DATE = 20100101
    DB_KEY = 'zx_industry_index'
    
    def get_data(self , date : int , end_date : int | None = None):
        if end_date is None:
            end_date = date
        df = self.iterate_fetch(self.pro.ci_daily , limit = 5000 , start_date = str(date) , end_date = str(end_date))
        if not df.empty:
            df['trade_date'] = df['trade_date'].astype(int)
        return df

    def get_zx_index_quotes(self , start_date : int , end_date : int):
        """get zx index quotes"""
        date_dfs : dict[Any , pd.DataFrame] = {}
        index_dfs : dict[Any , pd.DataFrame] = {}
        data = self.get_data(start_date , end_date)
        if data.empty:
            return date_dfs , index_dfs
        for date , df in data.groupby('trade_date' , group_keys = True):
            date_dfs[date] = df
        for index , df in data.groupby('ts_code' , group_keys = True):
            index_dfs[index] = df
        return date_dfs , index_dfs

    def update_dates(self , dates , step = 25 , **kwargs) -> None:
        """update the fetcher given dates"""
        if self.check_server_down(): 
            return
        if not self.db_by_name:
            assert None not in dates , f'{self.__class__.__name__} use date type but date is None'
        assert step > 0 , f'step must be larger than 0 , got {step}'
        si = dates[np.arange(len(dates))[::step]]
        ei = dates[np.arange(len(dates))[step-1::step]]
        if len(si) != len(ei):
            ei = np.concatenate([ei , dates[-1:]])

        for start , end in zip(si , ei): 
            date_dfs , index_dfs = self.get_zx_index_quotes(start , end)
            for date , df in date_dfs.items():
                DB.save(df , self.DB_SRC , self.DB_KEY , date = date , verbose = True)
            for index , df in index_dfs.items():
                self.update_index_daily_file(index , df , verbose = False)

    def update_index_daily_file(self , index : str , df : pd.DataFrame , verbose = False):
        df_old = DB.load('index_daily_ts' , index , verbose = False)
        if not df_old.empty:
            df_old['trade_date'] = df_old['trade_date'].astype(int)
            df = pd.concat([df_old , df]).drop_duplicates('trade_date' , keep = 'last')
        df = df.sort_values('trade_date').reset_index(drop = True)
        DB.save(df , 'index_daily_ts' , index , verbose = verbose)
    
class THSConcept(MonthFetcher):
    """Tonghuashun Concept"""
    DB_SRC = 'membership_ts'
    DB_KEY = 'concept'
    def get_data(self , date : int):
        df_theme = pd.concat([self.pro.ths_index(exchange = 'A', type = 'N') , 
                              self.pro.ths_index(exchange = 'A', type = 'TH')]).reset_index(drop=True)
        dfs = []
        for i , ts_code in enumerate(df_theme['ts_code']):
            # Logger.stdout(i , ts_code)
            df = self.pro.ths_member(ts_code = ts_code)
            dfs.append(df)
        df_all = pd.concat([d for d in dfs if not d.empty]).rename(columns={'name':'concept'})
        df_all = df_all.merge(df_theme , on = 'ts_code' , how='left').rename(columns={'ts_code':'index_code'})
        df_all = ts_code_to_secid(df_all , 'code')
        df = df_all.reset_index(drop = True)
        return df
    
class CSI300Weight(RollingFetcher):
    """CSI 300 Weight"""
    START_DATE = 20041231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi300'
    INDEX_CODE = '399300.SZ'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False

    def get_data(self , start_dt , end_dt):
        return index_weight_get_data(self , self.INDEX_CODE , start_dt , end_dt)
        
class CSI500Weight(RollingFetcher):
    """CSI 500 Weight"""
    START_DATE = 20041231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi500'
    INDEX_CODE = '000905.SH'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False

    def get_data(self , start_dt , end_dt):
        return index_weight_get_data(self , self.INDEX_CODE , start_dt , end_dt)
    
class CSI800Weight(RollingFetcher):
    """CSI 800 Weight"""
    START_DATE = 20041231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi800'
    INDEX_CODE = '000906.SH'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False
    def get_data(self , start_dt , end_dt):
        return index_weight_get_data(self , self.INDEX_CODE , start_dt , end_dt)

class CSI1000Weight(RollingFetcher):
    """CSI 1000 Weight"""
    START_DATE = 20041231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi1000'
    INDEX_CODE = '000852.SH'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False
    def get_data(self , start_dt , end_dt):
        return index_weight_get_data(self , self.INDEX_CODE , start_dt , end_dt , limit = 4000)
    
class CSI2000Weight(RollingFetcher):
    """CSI 2000 Weight"""
    START_DATE = 20131231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi2000'
    INDEX_CODE = '932000.CSI'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False
    def get_data(self , start_dt , end_dt):
        return index_weight_get_data(self , self.INDEX_CODE , start_dt , end_dt , limit = 4000)
