import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Any , Literal , Optional

from .core import DataBlock
from .tushare import TSData
from ..basic import PATH , CONF , SILENT
from ..basic.util import Timer
from ..func.singleton import singleton
from ..func.time import date_offset , today

class GetData:
    @classmethod
    def data_dates(cls , db_src , db_key , start_dt : Optional[int] = None , end_dt : Optional[int] = None):
        dates = PATH.db_dates(db_src , db_key , start_dt= start_dt , end_dt=end_dt)
        return dates

    @classmethod
    def trade_dates(cls , start_dt : int = -1 , end_dt : int = 99991231):
        with SILENT:
            calendar = TSData.CALENDAR.calendar[TSData.CALENDAR.calendar['trade'] == 1].index.to_numpy()
            calendar = calendar[(calendar >= start_dt) & (calendar <= end_dt)]
        return calendar
    
    @classmethod
    def stocks(cls , listed = True , exchange = ['SZSE', 'SSE', 'BSE']):
        with SILENT:
            stocks = TSData.INFO.get_desc(set_index=False)
            if listed: stocks = stocks[stocks['list_dt'] > 0]
            if exchange: stocks = stocks[stocks['exchange_name'].isin(exchange)]
        return stocks.reset_index()
    
    @classmethod
    def st_stocks(cls):
        with SILENT:
            st = TSData.INFO.get_st()
        return st
    
    @classmethod
    def day_quote(cls , date : int) -> pd.DataFrame | None:
        with SILENT:
            q = PATH.db_load('trade_ts' , 'day' , date)[['secid','adjfactor','close','vwap']]
        return q
    
    @classmethod
    def daily_returns(cls , start_dt : int , end_dt : int):
        with SILENT:
            pre_start_dt = int(date_offset(start_dt , -20))
            feature = ['close' , 'vwap']
            block = BlockLoader('trade_ts' , 'day' , ['close' , 'vwap' , 'adjfactor']).load_block(pre_start_dt , end_dt).as_tensor()
            block = block.adjust_price().align_feature(feature)
            values = block.values[:,1:] / block.values[:,:-1] - 1
            secid  = block.secid
            date   = block.date[1:]
            new_date = block.date_within(start_dt , end_dt)

            block = DataBlock(values , secid , date , feature).align_date(new_date)
        return block

@dataclass(slots=True)
class BlockLoader:
    db_src  : str
    db_key  : str | list
    feature : Optional[list] = None

    def __post_init__(self):
        assert f'DB_{self.db_src}' in [p.name for p in PATH.database.iterdir()] , f'DB_{self.db_src} not in {PATH.database}'
        src_path = PATH.database.joinpath(f'DB_{self.db_src}')
        assert np.isin(self.db_key , [p.name for p in src_path.iterdir()]).all() , f'{self.db_key} not all in {src_path}'

    def load(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None , 
             parallel : Literal['thread' , 'process'] | None = 'thread' , max_workers = 20):
        return self.load_block(start_dt , end_dt , parallel = parallel , max_workers = max_workers)
    
    def load_block(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None , 
                   parallel : Literal['thread' , 'process'] | None = 'thread' , max_workers = 20):
        if end_dt is not None   and end_dt < 0:   end_dt   = today(end_dt)
        if start_dt is not None and start_dt < 0: start_dt = today(start_dt)

        sub_blocks = []
        db_keys = self.db_key if isinstance(self.db_key , list) else [self.db_key]
        for db_key in db_keys:
            with Timer(f' --> {self.db_src} blocks reading [{db_key}] DataBase'):
                blk = DataBlock.load_db(self.db_src , db_key , start_dt , end_dt , self.feature , 
                                        parallel = parallel , max_workers = max_workers)
                sub_blocks.append(blk)
        if len(sub_blocks) <= 1:  
            block = sub_blocks[0]
        else:
            with Timer(f' --> {self.db_src} blocks merging ({len(sub_blocks)})'): 
                block = DataBlock.merge(sub_blocks)
        return block
    
@dataclass(slots=True)
class FrameLoader:
    db_src  : str
    db_key  : str

    def __post_init__(self):
        assert PATH.database.joinpath(f'DB_{self.db_src}' , self.db_key).exists() , \
            f'{PATH.database}/{self.db_src}/{self.db_key} not exists'
    
    def load(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None , 
             parallel : Literal['thread' , 'process'] | None = 'thread' , max_workers = 20):
        return self.load_frame(start_dt , end_dt , parallel , max_workers)

    def load_frame(self , start_dt : Optional[int] = None , end_dt : Optional[int] = None , 
                   parallel : Literal['thread' , 'process'] | None = 'thread' , max_workers = 20):
        dates = GetData.data_dates(self.db_src , self.db_key , start_dt , end_dt)
        dfs = PATH.db_load_multi(self.db_src , self.db_key , dates , parallel = parallel, max_workers=max_workers)
        dfs = [df.assign(date = date) for date,df in dfs.items() if df is not None and not df.empty]
        return pd.concat(dfs) if len(dfs) else pd.DataFrame()
    

@singleton
class DataVendor:
    '''
    Vender for most factor / portfolio analysis related data
    '''
    
    def __init__(self):
        self.start_dt = 99991231
        self.end_dt   = -1
        self.max_date   = GetData.data_dates('trade_ts' , 'day')[-1]
        self.trade_date = GetData.trade_dates()
        self.all_stocks = GetData.stocks().sort_values('secid')
        self.st_stocks  = GetData.st_stocks()
        self.day_quotes : dict[int,pd.DataFrame] = {}
        self.last_quote_dt = self.file_dates('trade_ts','day').max()

    def secid(self , date : int | None = None): 
        stk = self.all_stocks
        if date is not None: stk = stk[(stk.list_dt <= date) & (stk.delist_dt > date)]
        return stk.secid.unique()

    @staticmethod
    def single_file(db_src , db_key , date : int | None = None):
        return PATH.db_load(db_src , db_key , date)
    
    @staticmethod
    def file_dates(db_src , db_key , start_dt : int | None = None , end_dt : int | None = None , year : int | None = None):
        return PATH.db_dates(db_src , db_key , start_dt=start_dt , end_dt=end_dt , year = year)

    def td_within(self , start_dt : int = -1 , end_dt : int = 99991231 , step : int = 1):
        return self.trade_date[(self.trade_date >= start_dt) & (self.trade_date <= end_dt)][::step]

    def td_offset(self , date , offset : int = 0) -> int | np.ndarray | Any:
        if np.isscalar(date):
            # assert isinstance(date , int) , date
            if date not in self.trade_date: date = self.trade_date[self.trade_date <= date][-1]
            if offset: date = self.trade_date[np.where(self.trade_date == date)[0][0] + offset]
            return date
        else:
            return np.array([self.td_offset(d , offset) for d in date])
        
    def td_next(self , date) -> int | np.ndarray | Any:
        return self.td_offset(date , 1)
    
    def td_prev(self , date) -> int | np.ndarray | Any:
        return self.td_offset(date , -1)
    
    def latest_td(self , date : int): return self.td_offset(date)

    def random_factor(self , start_dt = 20240101 , end_dt = 20240531 , step = 5 , nfactor = 2):
        date  = self.td_within(start_dt , end_dt , step)
        secid = self.secid()
        factor_val = DataBlock(np.random.randn(len(secid),len(date),1,nfactor),
                               secid,date,[f'factor{i+1}' for i in range(nfactor)])
        return factor_val

    def get_returns(self , start_dt : int , end_dt : int):
        td_within = self.td_within(start_dt , end_dt)
        if (not hasattr(self , 'day_ret')) or (not np.isin(td_within , self.day_ret.date).all()):
            self.day_ret  = GetData.daily_returns(start_dt , end_dt)

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
        if len(td_within) == 0: return
        with SILENT:
            early_dates , late_dates = self.update_dates(data_key , td_within)
            datas : list[DataBlock] = []
            if len(early_dates): datas.append(BlockLoader(db_src , db_key).load_block(early_dates.min() , early_dates.max()))
            if len(late_dates):  datas.append(BlockLoader(db_src , db_key).load_block(late_dates.min() , late_dates.max()))
            if hasattr(self , data_key): datas.append(getattr(self , data_key))
        return DataBlock.merge(datas)

    def get_quote(self , start_dt : int , end_dt : int):
        db_src , db_key = 'trade_ts' , 'day'
        data = self.get_named_data_block(start_dt , end_dt , db_src , db_key , 'daily_quote')
        if isinstance(data , DataBlock):
            self.daily_quote = data
        return

    def get_risk_exp(self , start_dt : int , end_dt : int):
        db_src , db_key = 'models' , 'risk_exp'
        data = self.get_named_data_block(start_dt , end_dt , db_src , db_key , 'tushare_cne5_exp')
        if isinstance(data , DataBlock):
            self.risk_exp = data
        return
    
    def get_quote_df(self , date : int | Any):
        d = date[0] if isinstance(date , np.ndarray) else date
        assert d > 0 , 'Attempt to get unaccessable date'
        if d not in self.day_quotes: 
            df = GetData.day_quote(d)
            if df is not None: self.day_quotes[d] = df
        return self.day_quotes.get(d , None)
    
    def get_quote_ret(self , date0 , date1):
        q0 = self.get_quote_df(date0)
        q1 = self.get_quote_df(date1)
        if q0 is None or q1 is None: return None
        q = q0.merge(q1 , on = 'secid')
        q['ret'] = q['close_y'] * q['adjfactor_y'] / q['close_x'] / q['adjfactor_x'] - 1
        q['ret_vwap'] = q['vwap_y'] * q['adjfactor_y'] / q['vwap_x'] / q['adjfactor_x'] - 1
        q = q[['secid' , 'ret' , 'ret_vwap']].set_index('secid')
        return q

    def nday_fut_ret(self , secid : np.ndarray , date : np.ndarray , nday : int = 10 , lag : int = 2 , 
                     ret_type : Literal['close' , 'vwap'] = 'close'):
        assert lag > 0 , f'lag must be positive : {lag}'
        date_min = int(self.td_offset(date.min() , -10))
        date_max = int(self.td_offset(int(date.max()) , nday + lag + 10))
        self.get_returns(date_min , date_max)
        full_date = self.td_within(date_min , date_max)

        block = self.day_ret.align(secid , full_date , [ret_type] , inplace=False).as_tensor()
        block.values = F.pad(block.values[:,lag:] , (0,0,0,0,0,lag) , value = torch.nan)

        new_value = block.values.unfold(1 , nday , 1).exp().prod(dim = -1) - 1
        feature   = ['ret']

        new_block = DataBlock(new_value , secid , full_date[:new_value.shape[1]] , feature).align_date(date)
        return new_block
    
    def risk_style_exp(self , secid : np.ndarray , date : np.ndarray):
        self.get_risk_exp(date.min() , date.max())
        block = self.risk_exp.align(secid , date , CONF.RISK_STYLE , inplace=False).as_tensor()
        return block
    
    def risk_industry_exp(self , secid : np.ndarray , date : np.ndarray):
        self.get_risk_exp(date.min() , date.max())
        block = self.risk_exp.align(secid , date , CONF.RISK_INDUS , inplace=False).as_tensor()
        return block
    
    def get_ffmv(self , secid : np.ndarray , d : int):
        if d not in self.trade_date: return None
        self.get_risk_exp(d , d)
        value = self.risk_exp.loc(secid = secid , date = d , feature = 'weight').flatten()
        return value
    
    def get_cp(self , secid : np.ndarray , d : int):
        if d not in self.trade_date: return None
        self.get_quote(d , d)
        value = self.daily_quote.loc(secid = secid , date = d , feature = 'close').flatten()
        return value
    
DATAVENDOR = DataVendor()