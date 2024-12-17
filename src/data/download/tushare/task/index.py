import pandas as pd

from src.data.download.tushare.basic import MonthFetcher , RollingFetcher , pro , code_to_secid

def index_weight_get_data(self : RollingFetcher , index_code , start_date , end_date , limit = 4000):
    assert start_date is not None and end_date is not None , 'start_date and end_date must be provided'
    df = self.iterate_fetch(pro.index_weight , limit = limit , max_fetch_times = 500 , index_code=index_code , 
                            start_date = str(start_date) , end_date = str(end_date))
    if df.empty: return df
    df = code_to_secid(df , 'con_code').rename(columns={'trade_date':self.ROLLING_DATE_COL})
    df = df.sort_values([self.ROLLING_DATE_COL ,'weight'] , ascending=False)
    df['weight'] = df.groupby(self.ROLLING_DATE_COL)['weight'].transform(lambda x: x / x.sum())
    return df

class THSConcept(MonthFetcher):
    '''Tonghuashun Concept'''
    DB_SRC = 'membership_ts'
    DB_KEY = 'concept'
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
    
class CSI300Weight(RollingFetcher):
    START_DATE = 20041231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi300'
    INDEX_CODE = '399300.SZ'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False

    def get_data(self , start_date , end_date):
        return index_weight_get_data(self , self.INDEX_CODE , start_date , end_date)
        
class CSI500Weight(RollingFetcher):
    START_DATE = 20041231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi500'
    INDEX_CODE = '000905.SH'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False

    def get_data(self , start_date , end_date):
        return index_weight_get_data(self , self.INDEX_CODE , start_date , end_date)
    
class CSI800Weight(RollingFetcher):
    START_DATE = 20041231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi800'
    INDEX_CODE = '000906.SH'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False
    def get_data(self , start_date , end_date):
        return index_weight_get_data(self , self.INDEX_CODE , start_date , end_date)
    
class CSI1000Weight(RollingFetcher):
    START_DATE = 20041231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi1000'
    INDEX_CODE = '000852.SH'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False
    def get_data(self , start_date , end_date):
        return index_weight_get_data(self , self.INDEX_CODE , start_date , end_date , limit = 4000)
    
class CSI2000Weight(RollingFetcher):
    START_DATE = 20131231
    DB_SRC = 'benchmark_ts'
    DB_KEY = 'csi2000'
    INDEX_CODE = '932000.CSI'
    ROLLING_SEP_DAYS = 250
    ROLLING_BACK_DAYS = 10
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = False
    def get_data(self , start_date , end_date):
        return index_weight_get_data(self , self.INDEX_CODE , start_date , end_date , limit = 4000)