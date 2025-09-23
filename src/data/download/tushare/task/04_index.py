# do not use relative import in this file because it will be running in top-level directory
import pandas as pd

from src.data.download.tushare.basic import InfoFetcher , MonthFetcher , RollingFetcher , ts_code_to_secid

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

        df = pd.concat(dfs)
        return df
    
class THSConcept(MonthFetcher):
    """Tonghuashun Concept"""
    DB_SRC = 'membership_ts'
    DB_KEY = 'concept'
    def get_data(self , date : int):
        df_theme = pd.concat([self.pro.ths_index(exchange = 'A', type = 'N') , 
                              self.pro.ths_index(exchange = 'A', type = 'TH')]).reset_index(drop=True)
        dfs = []
        for i , ts_code in enumerate(df_theme['ts_code']):
            # print(i , ts_code)
            df = self.pro.ths_member(ts_code = ts_code)
            dfs.append(df)
        df_all = pd.concat(dfs).rename(columns={'name':'concept'})
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