import torch
import numpy as np
import pandas as pd

from typing import Any , Literal

from src.proj import Silence , Logger
from src.basic import CALENDAR , CONF , DB
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
        with Silence():
            self.all_stocks = self.INFO.get_desc(set_index=False , listed = listed , exchange = exchange)
            self.st_stocks = self.INFO.get_st()

    def secid(self , date : int | None = None) -> np.ndarray: 
        if date is None: 
            return np.unique(self.all_stocks['secid'].to_numpy(int))
        if date not in self.day_secids:
            self.day_secids[date] = self.INFO.get_secid(date)
        return self.day_secids[date]
    
    @classmethod
    def td_within(cls , start_dt : int | None = None , end_dt : int | None = None , step : int = 1 , updated = False , extend = 0):
        if extend > 0:
            if end_dt is not None:
                end_dt = cls.cd(end_dt , extend)
            if start_dt is not None:
                start_dt = cls.cd(start_dt , -extend)
        dates = CALENDAR.td_within(start_dt , end_dt , step , updated = updated)
        return dates
    
    @staticmethod
    def td_array(date , offset : int = 0): return CALENDAR.td_array(date , offset)
    
    @staticmethod
    def td(date , offset : int = 0): return CALENDAR.td(date , offset).as_int()

    @staticmethod
    def cd(date , offset : int = 0): return CALENDAR.cd(date , offset)

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
            Logger.alert(f'EmptyData: None of {names} found in {start_dt} ~ {end_dt}' , level = 1)
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

    def update_named_data_block(
        self,
        data_key: Literal["daily_quotes", "risk_exp"],
        db_src: str,
        db_key: str,
        dates: np.ndarray | list[int] | int | None = None,
        extend=0,
    ):
        if dates is None or (not isinstance(dates , int) and len(dates) == 0):
            return
        if isinstance(dates , int):
            target_start , target_end = dates , dates
        elif isinstance(dates , np.ndarray):
            target_start , target_end = dates.min() , dates.max()
        elif isinstance(dates , list):
            target_start , target_end = min(dates) , max(dates)
        else:
            raise ValueError(f'Unknown dates type: {type(dates)}')
        target_start , target_end = self.cd(target_start , -extend) , self.cd(target_end , extend)

        block0 : DataBlock = getattr(self , f'_block_{data_key}' , DataBlock())
        loaded_start , loaded_end = block0.min_date , block0.max_date
        
        if loaded_start <= target_start and loaded_end >= target_end:
            return
        target_start , target_end = min(target_start , loaded_start), max(target_end , loaded_end)
        dates = self.td_within(target_start , target_end)

        if len(early_dates := dates[dates < block0.min_date]) > 0:
            block = BlockLoader(db_src , db_key).load(early_dates.min() , early_dates.max() , silent = True).adjust_price()
            block0 = block0.merge_others(block , inplace = True)
        if len(late_dates := dates[dates > block0.max_date]) > 0:
            block = BlockLoader(db_src , db_key).load(late_dates.min() , late_dates.max() , silent = True).adjust_price()
            block0 = block0.merge_others(block , inplace = True)
        if not Silence.silent:
            Logger.success(f'DATAVENDOR: {data_key} expand from {loaded_start} ~ {loaded_end} to {target_start} ~ {target_end}')
        setattr(self , f'_block_{data_key}' , block0)

    def update_return_block(self , start_dt : int , end_dt : int):
        td_within = self.td_within(start_dt , end_dt , updated = True)
        daily_ret = getattr(self , f'_block_daily_ret' , DataBlock())
        if daily_ret.date is None or not np.isin(td_within , daily_ret.date).all():
            pre_start_dt = CALENDAR.cd(start_dt , -20)
            extend_td_within = self.td_within(pre_start_dt , end_dt)
            blk = self.get_quotes_block(extend_td_within).align(date = extend_td_within , feature = ['close' , 'vwap']).as_tensor().ffill()
            blk.update(values = torch.nn.functional.pad(blk.values[:,1:] / blk.values[:,:-1] - 1 , (0,0,0,0,1,0) , value = torch.nan))
            blk = blk.align_date(blk.date_within(start_dt , end_dt) , inplace = True)
            setattr(self , f'_block_daily_ret' , blk)

    def get_quotes_block(self , dates : np.ndarray | list[int] | int | None = None , extend = 0) -> DataBlock:
        with Silence(True):
            self.update_named_data_block('daily_quotes' , 'trade_ts' , 'day' , dates , extend)
        return getattr(self , f'_block_daily_quotes' , DataBlock())

    def get_risk_exp(self , dates : np.ndarray | list[int] | int | None = None , extend = 0) -> DataBlock:
        with Silence(True):
            self.update_named_data_block('risk_exp' , 'models' , 'tushare_cne5_exp' , dates , extend)
        return getattr(self , f'_block_risk_exp' , DataBlock())

    def get_returns_block(self , start_dt : int , end_dt : int):
        with Silence(True):
            self.update_return_block(start_dt , end_dt)
        return getattr(self , f'_block_daily_ret' , DataBlock())
    
    def day_quote(self , date : int | Any , price : Literal['close' , 'vwap' , 'open'] = 'close'):
        df = self.TRADE.get_trd(date , ['secid' , 'adjfactor' , price])
        if not df.empty:
            df['price'] = df[price] * df['adjfactor'].fillna(1)
            return df.loc[:,['secid' , 'price']]
        else:
            return pd.DataFrame(columns = ['secid' , 'price'])
    
    def get_quote_ret(self , date0 , date1 , 
                      price0 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      price1 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      secid : np.ndarray | pd.Series | Any | None = None):
        """
        get ret of single date0 and date1
        using DataFrame method is much faster than DataBlock method
        slicing of smaller df is much faster than slicing of larger array
        """
        q0 = self.day_quote(date0 , price0)
        q1 = self.day_quote(date1 , price1)
        if q0.empty or q1.empty: 
            return pd.DataFrame(columns = ['secid' , 'ret']).set_index('secid')
        q = q0.query('price != 0').merge(q1 , on = 'secid')
        q['ret'] = q['price_y'] / q['price_x'] - 1
        q = q[['secid' , 'ret']].set_index('secid')
        if secid is not None:
            q = q.reindex(secid).fillna(0)
        return q

    def get_quote_ret_new(self , date0 , date1 , 
                      price0 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      price1 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      secid : np.ndarray | pd.Series | Any | None = None):
        blk = self.get_quotes_block([date0 , date1])
        p0 = blk.loc(date = date0 , feature = price0).flatten()
        p1 = blk.loc(date = date1 , feature = price1).flatten()
        if len(p0) == 0 or len(p1) == 0:
            return pd.DataFrame(columns = ['secid' , 'ret']).set_index('secid')
        q = pd.DataFrame({'secid' : blk.secid , 'ret' : p1 / np.where(p0 == 0 , np.nan , p0) - 1}).set_index('secid')
        if secid is not None:
            q = q.reindex(secid).fillna(0)
        # q1 = self.get_quote_ret_old(date0 , date1 , price0 , price1 , secid)
        # q2 = q1.merge(q , on = 'secid' , how = 'left').query('abs(ret_x - ret_y) > 1e-5')
        # if len(q2) > 0:
        #     Logger.stdout(q2)
        #     raise Exception('stop')
        return q

    def get_miscel_ret(self , df : pd.DataFrame , ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        """get ret of miscel secids and dates, df must contain 'secid' , 'start' , 'end' columns"""
        assert 'secid' in df.columns and 'start' in df.columns and 'end' in df.columns , \
            f'df must contain "secid" , "start" , "end" columns : {df.columns}'
        df['prev'] = CALENDAR.td_array(df['start'] , -1)
        dates = np.unique(np.concatenate([df['prev'].to_numpy() , df['end'].to_numpy()]))
        quotes = DB.load_multi('trade_ts' , 'day' , dates).filter(items = ['secid' , 'date' , ret_type , 'adjfactor'])
        quotes[ret_type] = quotes[ret_type] * quotes['adjfactor']

        q0 = df.merge(quotes , left_on = ['secid' , 'prev'] , right_on = ['secid' , 'date'] , how = 'left')[ret_type]
        q1 = df.merge(quotes , left_on = ['secid' , 'end'] , right_on = ['secid' , 'date'] , how = 'left')[ret_type]
        ret = q1 / q0 - 1
        df['ret'] = ret
        del df['prev']
        return df

    def nday_fut_ret(self , secid : np.ndarray , date : np.ndarray , nday : int = 10 , lag : int = 2 , 
                     ret_type : Literal['close' , 'vwap'] = 'close'):
        assert lag > 0 , f'lag must be positive : {lag}. If you want to use next day\'s return, set lag = 1'
        date_min = self.td(date.min() , -10)
        date_max = self.td(int(date.max()) , nday + lag + 10)
        full_date = self.td_within(date_min , date_max)
        blk = self.get_returns_block(date_min , date_max).align(secid , full_date , ret_type).as_tensor()
        values = torch.nn.functional.pad(blk.values[:,lag:] , (0,0,0,0,0,lag) , value = torch.nan).unfold(1 , nday , 1).exp().prod(dim = -1) - 1
        blk.update(values = values , date = full_date[:values.shape[1]] , feature = ['ret']).align_date(date , inplace = True)
        return blk
    
    def ffmv(self , secid : np.ndarray , date : np.ndarray , prev = True):
        if prev : 
            date = self.td_array(date , -1)
        blk = self.get_risk_exp(date).align(secid , date , ['weight'])
        if prev : 
            blk.date = self.td_array(blk.date , 1)
        return blk
    
    def risk_style_exp(self , secid : np.ndarray , date : np.ndarray):
        blk = self.get_risk_exp(date).align(secid , date , CONF.Factor.RISK.style)
        return blk
    
    def risk_industry_exp(self , secid : np.ndarray , date : np.ndarray):
        blk = self.get_risk_exp(date).align(secid , date , CONF.Factor.RISK.indus)
        return blk
    
    def get_ffmv(self , secid : np.ndarray , d : int):
        if not CALENDAR.is_trade_date(d): 
            return None
        blk = self.get_risk_exp(d)
        value = blk.loc(secid = secid , date = d , feature = 'weight').flatten()
        return value
    
    def get_cp(self , secid : np.ndarray , d : int):
        if not CALENDAR.is_trade_date(d): 
            return None
        blk = self.get_quotes_block(d)
        value = blk.loc(secid = secid , date = d , feature = 'close').flatten()
        return value

    def get_fin_latest(self , expression : str , date : int , new_name : str | None = None , **kwargs) -> pd.Series:
        '''statement@field@fin_type'''
        fin_data = FinData(expression , **kwargs)
        return fin_data.get_latest(date , new_name)

    def get_fin_hist(self, expression : str , date : int , lastn : int , new_name : str | None = None , **kwargs) -> pd.DataFrame:
        '''statement@field@fin_type'''
        fin_data = FinData(expression , **kwargs)
        return fin_data.get_hist(date , lastn , new_name)

    def get_fin_qoq(self , expression : str , date : int , lastn : int , method : Literal['pct' , 'diff'] = 'pct' , **kwargs) -> pd.DataFrame:
        data = FinData(expression , **kwargs).get_hist(date = date , lastn = lastn + 2)
        full_index = pd.MultiIndex.from_product([data.index.get_level_values('secid').unique() ,
                                                 data.index.get_level_values('end_date').unique()])
        df_yoy = data.reindex(full_index)
        df_yoy_base = df_yoy.groupby('secid').shift(1)
        df_yoy = (df_yoy - df_yoy_base) 
        if method == 'pct':
            df_yoy = df_yoy / df_yoy_base.abs()

        df_yoy = df_yoy.reindex(data.index).where(~data.isna() , np.nan).replace([np.inf , -np.inf] , np.nan)
        return df_yoy.groupby('secid').tail(lastn)
    
    def get_fin_yoy(self , expression : str , date : int , lastn : int , method : Literal['pct' , 'diff'] = 'pct' , **kwargs) -> pd.DataFrame:
        data = FinData(expression , **kwargs).get_hist(date = date , lastn = lastn + 5)
        full_index = pd.MultiIndex.from_product([data.index.get_level_values('secid').unique() ,
                                                 data.index.get_level_values('end_date').unique()])
        df_yoy = data.reindex(full_index)
        df_yoy_base = df_yoy.groupby('secid').shift(4)
        df_yoy = (df_yoy - df_yoy_base) 
        if method == 'pct':
            df_yoy = df_yoy / df_yoy_base.abs()

        df_yoy = df_yoy.reindex(data.index).where(~data.isna() , np.nan).replace([np.inf , -np.inf] , np.nan)
        return df_yoy.groupby('secid').tail(lastn)
    
DATAVENDOR = DataVendor()