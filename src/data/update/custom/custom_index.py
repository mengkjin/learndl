import pandas as pd
import numpy as np

from abc import ABCMeta , abstractmethod
from typing import Literal , Type , Any
from src.proj import Logger , CALENDAR , DB
from src.data.loader.data_vendor import DATAVENDOR
from src.data.update.custom.basic import BasicCustomUpdater

START_DATE = 20100101
DB_SRC = 'index_daily_custom'

class CustomIndexMeta(ABCMeta):
    registry : dict[str , Type['CustomIndex'] | Any] = {}
    def __new__(cls , name , bases , dct):
        new_cls = super().__new__(cls , name , bases , dct)
        assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
            f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
        if not new_cls.__abstractmethods__:
            cls.registry[name] = new_cls
        return new_cls

class CustomIndexName:
    def __get__(self , instance , owner):
        return owner.__name__.lower()

class CustomIndexPortClass:
    def __get__(self , instance , owner):
        if not hasattr(self , '_Port'):
            from src.res.factor.util.classes.port import Port
            self._Port = Port
        return self._Port

class CustomIndex(metaclass=CustomIndexMeta):
    start_date : int = START_DATE
    index_name = CustomIndexName()
    PortClass = CustomIndexPortClass()

    @abstractmethod
    def rebalance_dates(self) -> np.ndarray:
        """get rebalance dates"""

    @abstractmethod
    def rebalance_portfolio(self , date : int) -> pd.DataFrame:
        """get index portfolio"""

    @property
    def reb_dates(self) -> np.ndarray:
        if not hasattr(self , '_reb_dates'):
            self._reb_dates = self.rebalance_dates()
        return self._reb_dates

    @property
    def target_path(self):
        return DB.path(DB_SRC , self.index_name)

    @property
    def current_portfolio(self):
        if not hasattr(self , '_current_portfolio'):
            return None
        else:
            return self._current_portfolio

    @current_portfolio.setter
    def current_portfolio(self , port):
        assert isinstance(port , self.PortClass) , f'port is not a {self.PortClass} instance: {type(port)}'
        self._current_portfolio = port

    def index_portfolio(self , date : int):
        """get index portfolio"""
        if self.current_portfolio is not None and self.current_portfolio.date == date:
            return self.current_portfolio
        reb_dates = self.reb_dates
        prev_reb_date = 99991231 if (len(reb_dates) == 0 or min(reb_dates) > date) else max(reb_dates[reb_dates <= date])
        if prev_reb_date > date:
            port = self.PortClass(pd.DataFrame() , date , self.index_name)
        elif prev_reb_date == date:
            port = self.PortClass(self.rebalance_portfolio(date) , date , self.index_name)
        elif self.current_portfolio is not None and self.current_portfolio.date >= prev_reb_date:
            port = self.current_portfolio.evolve_to_date(date)
        else:
            prev_reb_port = self.PortClass(self.rebalance_portfolio(prev_reb_date) , prev_reb_date , self.index_name)
            port = prev_reb_port.evolve_to_date(date)
        self.current_portfolio = port
        return self.current_portfolio

    def index_return(self , date : int) -> float:
        """get index return"""
        prev_date = CALENDAR.td(date , -1).td
        prev_port = self.index_portfolio(prev_date)
        if prev_port.is_emtpy():
            return 0.
        return prev_port.fut_ret(date)

    def update_dates(self , dates : np.ndarray | list[int] , indent : int = 1 , vb_level : int = 1):
        """update index return for given dates"""
        if len(dates) == 0:
            return
        dates = np.sort(CALENDAR.slice(dates , self.start_date , CALENDAR.updated()))
        pct_chg = np.array([self.index_return(date) for date in dates]) * 100
        df = pd.DataFrame({'trade_date' : dates , 'pct_chg' : pct_chg})
        old_df = DB.load_df(self.target_path)
        if not old_df.empty:
            df = pd.concat([old_df.query('trade_date not in @df.trade_date') , df]).sort_values('trade_date').reset_index(drop=True)
        DB.save_df(df , self.target_path , indent = indent , vb_level = vb_level)

    def stored_dates(self) -> np.ndarray:
        """get stored dates"""
        old_df = DB.load_df(self.target_path)
        if old_df.empty:
            return np.array([])
        return old_df['trade_date'].to_numpy(int)

    def target_dates(self , update_type : Literal['recalc' , 'update' , 'rollback'] , 
                     rollback_date : int = 99991231 , start_date : int = START_DATE , **kwargs):
        full_dates = CALENDAR.td_within(max(start_date , self.start_date) , CALENDAR.updated())
        if update_type == 'recalc':
            return full_dates
        elif update_type == 'update':
            return full_dates[~np.isin(full_dates , self.stored_dates())]
        elif update_type == 'rollback':
            return full_dates[full_dates >= rollback_date]
        else:
            raise ValueError(f'Invalid update type: {update_type}')

    @classmethod
    def iter_custom_indices(cls):
        for name , custom_index_cls in cls.registry.items():
            yield custom_index_cls()

class CustomIndexUpdater(BasicCustomUpdater):
    START_DATE = START_DATE

    @classmethod
    def custom_indices(cls):
        return list(CustomIndex.iter_custom_indices())
    
    @classmethod
    def update_all(cls , update_type : Literal['recalc' , 'update' , 'rollback'] , indent : int = 1 , vb_level : int = 1):
        if update_type == 'recalc':
            Logger.warning(f'Recalculate all custom index is supported , but beware of the performance!')

        total_dates = []
        custom_indices = list(CustomIndex.iter_custom_indices())
        for custom_index in custom_indices:
            target_dates = custom_index.target_dates(update_type , rollback_date = cls._rollback_date , start_date = cls.START_DATE)
            if len(target_dates) == 0:
                Logger.skipping(f'{custom_index.index_name} is up to date' , indent = indent + 1 , vb_level = vb_level + 2)
                continue
            custom_index.update_dates(target_dates , indent = indent + 1 , vb_level = vb_level + 2)
            total_dates.extend(target_dates.tolist())
        if total_dates == 0:
            Logger.skipping(f'All custom indices are up to date' , indent = indent , vb_level = vb_level)
        else:
            Logger.success(f'{len(custom_indices)} custom indices updated at {CALENDAR.dates_str(total_dates)}' , indent = indent , vb_level = vb_level)

    @classmethod
    def update_one(cls , date : int , indent : int = 2 , vb_level : int = 2):
        for custom_index in CustomIndex.iter_custom_indices():
            custom_index.update_dates([date] , indent = indent , vb_level = vb_level)

class MicroCap_400(CustomIndex):
    def rebalance_dates(self) -> np.ndarray:
        return CALENDAR.td_within(self.start_date , CALENDAR.updated())

    def rebalance_portfolio(self , date : int) -> pd.DataFrame:
        prev_date = CALENDAR.td(date , -1)
        prev_val = DATAVENDOR.TRADE.get_val(prev_date)
        prev_val['free_cap'] = prev_val['free_share'] * prev_val['close']
        df = prev_val.loc[:,['secid','free_cap']].reset_index(drop=True)
        pool = DATAVENDOR.INFO.get_desc(prev_date , exchange = ['SZSE', 'SSE'] , set_index=False)['secid']
        st_stocks = DATAVENDOR.INFO.get_st(prev_date)['secid']
        df = df[df['secid'].isin(pool) & ~df['secid'].isin(st_stocks)].sort_values('free_cap' , ascending=True).\
            head(400).reset_index(drop=True).drop(columns=['free_cap'])
        df['weight'] = 1 / len(df)
        return df