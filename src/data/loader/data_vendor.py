import torch
import numpy as np
import pandas as pd

from typing import Any , Literal

from src.basic import CALENDAR , CONF , SILENT , DB
from src.func.singleton import singleton
from src.data.util import DataBlock , INFO

from .loader import BlockLoader
from .financial_data import BS , IS , CF , INDI , FINA , FinData
from .analyst import ANALYST
from .min_kline import MKLINE    
from .model_data import RISK
from .trade_data import TRADE
from .exposure import EXPO

@singleton
class DataVendor:
    '''
    Vender for most factor / portfolio analysis related data
    '''
    CALENDAR = CALENDAR
    TRADE = TRADE
    INFO = INFO
    RISK = RISK
    
    INDI = INDI
    BS   = BS
    IS   = IS
    CF   = CF
    FINA = FINA

    ANALYST = ANALYST

    MKLINE = MKLINE
    EXPO = EXPO

    CUSTOM_DATA = {}
    
    def __init__(self):
        self.start_dt = 99991231
        self.end_dt   = -1

        dates = DB.dates('trade_ts' , 'day')
        self.max_date   = dates[-1] if len(dates) else -1
        self.min_date   = dates[0] if len(dates) else 99991231

        self.day_quotes : dict[int,pd.DataFrame] = {}
        self.day_secids : dict[int,np.ndarray] = {}
        self.last_quote_dt = self.max_date

        self.init_stocks()

    def data_storage_control(self):
        self.TRADE.truncate()
        self.RISK.truncate()
        
        self.INDI.truncate()
        self.BS.truncate()
        self.IS.truncate()
        self.CF.truncate()
        self.FINA.truncate()

    def init_stocks(self , listed = True , exchange = ['SZSE', 'SSE', 'BSE']):
        with SILENT:
            self.all_stocks = self.INFO.get_desc(set_index=False , listed = listed , exchange = exchange)
            self.st_stocks = self.INFO.get_st()

    def secid(self , date : int | None = None) -> np.ndarray: 
        if date is None: 
            return np.unique(self.all_stocks['secid'].to_numpy(int))
        if date not in self.day_secids:
            self.day_secids[date] = self.INFO.get_secid(date)
        return self.day_secids[date]
    
    @classmethod
    def td_within(cls , start_dt : int | None = None , end_dt : int | None = None , step : int = 1 , updated = False):
        dates = CALENDAR.td_within(start_dt , end_dt , step , updated = updated)
        return dates
    
    @staticmethod
    def td_array(date , offset : int = 0): return CALENDAR.td_array(date , offset)
    
    @staticmethod
    def td(date , offset : int = 0): return CALENDAR.td(date , offset)

    @classmethod
    def real_factor(cls , factor_type : Literal['pred' , 'factor'] , names : str | list[str] | np.ndarray , 
                    start_dt = 20240101 , end_dt = 20240531 , step = 5):
        if isinstance(names , str): 
            names = [names]
        dates = DATAVENDOR.td_within(start_dt , end_dt , step)

        values = [DB.load_multi(factor_type , name , dates , date_colname = 'date') for name in names]
        values = [v.set_index(['secid','date']) for v in values if not v.empty]
        if values:
            return DataBlock.from_dataframe(pd.concat(values , axis=1).sort_index())
        else:
            print(f'None of {names} found in {start_dt} ~ {end_dt}')
            return DataBlock()

    @classmethod
    def stock_factor(cls , factor_names : str | list[str] | np.ndarray , start_dt = 20240101 , end_dt = 20240531 , step = 5):
        return cls.real_factor('factor' , factor_names , start_dt , end_dt , step)
        
    @classmethod
    def model_preds(cls , model_names : str | list[str] | np.ndarray , start_dt = 20240101 , end_dt = 20240531 , step = 5):
        return cls.real_factor('pred' , model_names , start_dt , end_dt , step)
            
    def random_factor(self , start_dt = 20240101 , end_dt = 20240531 , step = 5 , nfactor = 2):
        date  = self.td_within(start_dt , end_dt , step)
        secid = self.secid()
        return DataBlock(np.random.randn(len(secid),len(date),1,nfactor),
                         secid,date,[f'factor{i+1}' for i in range(nfactor)])

    def get_returns(self , start_dt : int , end_dt : int):
        td_within = self.td_within(start_dt , end_dt , updated = True)
        if (not hasattr(self , 'day_ret')) or (not np.isin(td_within , self.day_ret.date).all()):
            with SILENT:
                pre_start_dt = CALENDAR.cd(start_dt , -20)
                feature = ['close' , 'vwap']
                loader = BlockLoader('trade_ts' , 'day' , ['close' , 'vwap' , 'adjfactor'])
                block : DataBlock | Any = loader.load_block(pre_start_dt , end_dt).as_tensor()
                block = block.adjust_price().align_feature(feature)
                values = block.values[:,1:] / block.values[:,:-1] - 1
                secid  = block.secid
                date   = block.date[1:]
                new_date = block.date_within(start_dt , end_dt)
                self.day_ret = DataBlock(values , secid , date , feature).align_date(new_date)
                
    def update_dates(self , data_key : str , dates : np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        if hasattr(self , data_key):
            exist_dates = getattr(self , data_key).date
            if len(exist_dates):
                early_dates = dates[dates < exist_dates.min()]
                late_dates  = dates[dates > exist_dates.min()]
            else:
                early_dates = dates
                late_dates  = np.array([])
        else:
            early_dates = dates
            late_dates  = np.array([])
        return early_dates , late_dates

    def get_named_data_block(self , start_dt : int , end_dt : int , db_src , db_key , data_key):
        td_within = self.td_within(start_dt , end_dt)
        if len(td_within) == 0: 
            return
        with SILENT:
            early_dates , late_dates = self.update_dates(data_key , td_within)
            datas : list[DataBlock] = []
            if len(early_dates): 
                datas.append(BlockLoader(db_src , db_key).load_block(early_dates.min() , early_dates.max()))
            if len(late_dates):  
                datas.append(BlockLoader(db_src , db_key).load_block(late_dates.min() , late_dates.max()))
            if hasattr(self , data_key): 
                datas.append(getattr(self , data_key))
        return DataBlock.merge(datas)

    def get_quote(self , start_dt : int , end_dt : int):
        db_src , db_key = 'trade_ts' , 'day'
        data = self.get_named_data_block(start_dt , end_dt , db_src , db_key , 'daily_quote')
        if isinstance(data , DataBlock):
            self.daily_quote = data
        return

    def get_risk_exp(self , start_dt : int , end_dt : int):
        db_src , db_key = 'models' , 'tushare_cne5_exp'
        data = self.get_named_data_block(start_dt , end_dt , db_src , db_key , 'tushare_cne5_exp')
        if isinstance(data , DataBlock):
            self.risk_exp = data
        return
    
    def get_quote_df(self , date : int | Any):
        d = date[0] if isinstance(date , np.ndarray) else date
        assert d > 0 , 'Attempt to get unaccessable date'
        if d not in self.day_quotes: 
            df : pd.DataFrame | Any = DB.load('trade_ts' , 'day' , date)[['secid','adjfactor','close','vwap','open']]
            if df is not None: 
                self.day_quotes[d] = df
        return self.day_quotes.get(d , None)
    
    def get_quote_ret(self , date0 , date1 , 
                      price0 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      price1 : Literal['close' , 'vwap' , 'open'] = 'close'):
        q0 = self.get_quote_df(date0)
        q1 = self.get_quote_df(date1)
        if q0 is None or q1 is None: 
            return None
        q = q0.merge(q1 , on = 'secid')
        q['adjfactor_x'] = q['adjfactor_x'].fillna(1)
        q['adjfactor_y'] = q['adjfactor_y'].fillna(1)
        
        q['ret'] = q[f'{price1}_y'] * q['adjfactor_y'] / q[f'{price0}_x'] / q['adjfactor_x'] - 1
        q = q[['secid' , 'ret']].set_index('secid')
        return q

    def get_miscel_ret(self , df : pd.DataFrame , ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        """get ret of miscel secids and dates, df must contain 'secid' , 'start' , 'end' columns"""
        assert 'secid' in df.columns and 'start' in df.columns and 'end' in df.columns , \
            f'df must contain "secid" , "start" , "end" columns : {df.columns}'
        dates = np.unique(np.concatenate([df['start'].to_numpy() , df['end'].to_numpy()]))
        quotes = DB.load_multi('trade_ts' , 'day' , dates).filter(items = ['secid' , 'date' , ret_type])

        df = df.merge(quotes.rename(columns = {'date' : 'start' , ret_type : 'p0'}) , on = ['secid' , 'start'] , how = 'left')
        df = df.merge(quotes.rename(columns = {'date' : 'end' , ret_type : 'p1'}) , on = ['secid' , 'end'] , how = 'left')
        df = df.assign(ret = df['p1'] / df['p0'] - 1)
        df = df.filter(items = ['secid' , 'start' , 'end' , 'ret'])
        return df

    def nday_fut_ret(self , secid : np.ndarray , date : np.ndarray , nday : int = 10 , lag : int = 2 , 
                     ret_type : Literal['close' , 'vwap'] = 'close'):
        assert lag > 0 , f'lag must be positive : {lag}. If you want to use next day\'s return, set lag = 1'
        date_min = self.td(date.min() , -10).td
        date_max = self.td(int(date.max()) , nday + lag + 10).td
        self.get_returns(date_min , date_max)
        full_date = self.td_within(date_min , date_max)

        block = self.day_ret.align(secid , full_date , [ret_type] , inplace=False).as_tensor()
        block.values = torch.nn.functional.pad(block.values[:,lag:] , (0,0,0,0,0,lag) , value = torch.nan)

        new_value = block.values.unfold(1 , nday , 1).exp().prod(dim = -1) - 1
        feature   = ['ret']

        new_block = DataBlock(new_value , secid , full_date[:new_value.shape[1]] , feature).align_date(date)
        return new_block
    
    def ffmv(self , secid : np.ndarray , date : np.ndarray , prev = True):
        if prev : 
            date = self.td_array(date , -1)
        self.get_risk_exp(date.min() , date.max())
        block = self.risk_exp.align(secid , date , ['weight'] , inplace=False).as_tensor()
        if prev : 
            block.date = self.td_array(block.date , 1)
        return block
    
    def risk_style_exp(self , secid : np.ndarray , date : np.ndarray):
        self.get_risk_exp(date.min() , date.max())
        block = self.risk_exp.align(secid , date , CONF.Factor.RISK.style , inplace=False).as_tensor()
        return block
    
    def risk_industry_exp(self , secid : np.ndarray , date : np.ndarray):
        self.get_risk_exp(date.min() , date.max())
        block = self.risk_exp.align(secid , date , CONF.Factor.RISK.indus , inplace=False).as_tensor()
        return block
    
    def get_ffmv(self , secid : np.ndarray , d : int):
        if not CALENDAR.is_trade_date(d): 
            return None
        self.get_risk_exp(d , d)
        value = self.risk_exp.loc(secid = secid , date = d , feature = 'weight').flatten()
        return value
    
    def get_cp(self , secid : np.ndarray , d : int):
        if not CALENDAR.is_trade_date(d): 
            return None
        self.get_quote(d , d)
        value = self.daily_quote.loc(secid = secid , date = d , feature = 'close').flatten()
        return value

    def get_fin_latest(self , expression : str , date : int , new_name : str | None = None , **kwargs) -> pd.Series:
        '''statement@field@fin_type'''
        fin_data = FinData(expression , **kwargs)
        return fin_data.get_latest(date , new_name)

    def get_fin_hist(self, expression : str , date : int , lastn : int , new_name : str | None = None , **kwargs) -> pd.DataFrame:
        '''statement@field@fin_type'''
        fin_data = FinData(expression , **kwargs)
        return fin_data.get_hist(date , lastn , new_name)

    def get_fin_qoq(self , expression : str , date : int , lastn : int , **kwargs) -> pd.DataFrame:
        data = FinData(expression , **kwargs).get_hist(date = date , lastn = lastn + 2)
        full_index = pd.MultiIndex.from_product([data.index.get_level_values('secid').unique() ,
                                                 data.index.get_level_values('end_date').unique()])
        df_yoy = data.reindex(full_index)
        df_yoy_base = df_yoy.groupby('secid').shift(1)
        df_yoy = (df_yoy - df_yoy_base) / df_yoy_base.abs()

        df_yoy = df_yoy.reindex(data.index).where(~data.isna() , np.nan).replace([np.inf , -np.inf] , np.nan)
        return df_yoy.groupby('secid').tail(lastn)
    
    def get_fin_yoy(self , expression : str , date : int , lastn : int , **kwargs) -> pd.DataFrame:
        data = FinData(expression , **kwargs).get_hist(date = date , lastn = lastn + 5)
        full_index = pd.MultiIndex.from_product([data.index.get_level_values('secid').unique() ,
                                                 data.index.get_level_values('end_date').unique()])
        df_yoy = data.reindex(full_index)
        df_yoy_base = df_yoy.groupby('secid').shift(4)
        df_yoy = (df_yoy - df_yoy_base) / df_yoy_base.abs()

        df_yoy = df_yoy.reindex(data.index).where(~data.isna() , np.nan).replace([np.inf , -np.inf] , np.nan)
        return df_yoy.groupby('secid').tail(lastn)
    
DATAVENDOR = DataVendor()