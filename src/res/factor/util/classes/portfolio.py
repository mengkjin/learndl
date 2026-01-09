import numpy as np
import pandas as pd

from copy import deepcopy
from pathlib import Path
from typing import Any , Literal

from src.proj import DB , CALENDAR
from src.data import DataBlock
from .port import Port

class Portfolio:
    """
    A portfolio is a collection of ports for multiple days.
    methods:
    .__len__() : return the number of dates
    .__bool__() : return True if the portfolio is not empty
    .__getitem__(date) : return the port for the given date
    .__setitem__(date , port) : set the port for the given date
    .copy() : return a copy of the portfolio
    .is_empty : return True if the portfolio is empty
    .name : return the lower-case name of the portfolio
    .port_date : return the dates of the ports
    .port_secid : return the secids of the ports
    .random() : generate a random portfolio with 3 random ports
    .weight_block() : generate the weight of the portfolio as a DataBlock
    .append(port , override = False , ignore_name = False) : append a port to the portfolio
    .available_dates() : return the available dates of the portfolio
    .latest_avail_date(date : int = 99991231) : return the latest available date of the portfolio before the given date
    .has(date : int) : return True if the portfolio has the given date
    .get(date : int , latest = False) : return the port for the given date
    .from_dataframe(df : pd.DataFrame , name : str | Any = None) : create a portfolio from a dataframe
    .to_dataframe() : convert the portfolio to a dataframe
    .load(path : Path | str) : load a portfolio from a path (dataframe)
    .save(path : Path | str , overwrite = False , append = True , indent : int = 1 , vb_level : int = 2) : save the portfolio to a path (dataframe)
    .filter_secid(secid : np.ndarray | Any | None = None , exclude = False , inplace = False) : filter the portfolio by secid , if exclude is True, filter out the secid, otherwise filter in the secid
    .filter_dates(dates : np.ndarray | Any | None = None , exclude = False , inplace = False) : filter the portfolio by dates , if exclude is True, filter out the dates, otherwise filter in the dates
    .replace(port : 'Port|Portfolio' , inplace = False) : replace the portfolio with the given port or portfolio
    .rename(new_name : str) : rename the portfolio
    .from_ports(*ports : Port , name : str | None = None) : create a portfolio from a list of ports
    .activate_accountant() : activate the accountant for the portfolio
    .accounting(benchmark : 'Portfolio | str | Any' = None , start : int = -1 , end : int = 99991231 , analytic = True , attribution = True , * , start_port : Port | None = None , trade_engine : Literal['default' , 'harvest' , 'yale'] | str = 'default' , daily = False , cache = False , indent : int = 0 , vb_level : int = 1) : account the portfolio
    .save_account(path : Path | str , overwrite = False , append = True , indent : int = 1 , vb_level : int = 2) : save the account to a path (a dir containing multiple dataframes)
    .load_account(path : Path | str) : load the account from a path (a dir containing multiple dataframes)
    .account : return the account of the portfolio
    .cached_accounts : return the cached accounts of the portfolio
    .activate_conditioner(name : str) : activate the conditioner for the portfolio
    .conditioned_portfolio(name : str) : return the conditioned portfolio
    .conditioned_pf_ret(name : str , plot = False) : return the conditioned portfolio return
    """

    def __init__(self , name : str | Any = None) -> None:
        self._name = self.get_object_name(name)
        self.ports : dict[int,Port] = {}
        self.is_default = name is None
        self.weight_block_completed = False
        self._last_port : Port | None = None
        
    def __len__(self): 
        """return the number of dates"""
        return len(self.available_dates())
    def __bool__(self): 
        """return True if the portfolio is not empty"""
        return len(self) > 0
    def __repr__(self): 
        return f'<{self.name}> : {len(self.ports)} ports'
    def __getitem__(self , date): 
        """return the port for the given date"""
        return self.get(date)
    def __setitem__(self , date , port): 
        """set the port for the given date"""
        assert date == port.date , (date , port.date)
        self.append(port , True)
    def copy(self): 
        """return a copy of the portfolio"""
        return deepcopy(self)
    @property
    def is_empty(self):
        """return True if the portfolio is empty"""
        return len(self) == 0
    @property
    def name(self): 
        """return the lower-case name of the portfolio"""
        return self._name.lower()

    @property
    def port_date(self): 
        """return the dates of the ports"""
        return np.array(list(self.ports.keys()) , dtype=int)
    @property
    def port_secid(self): 
        """return the secids of the ports"""
        return np.unique(np.concatenate([port.secid for port in self.ports.values()])) if self.ports else np.array([]).astype(int)

    @classmethod
    def random(cls):  
        """generate a random portfolio with 3 random ports"""
        rand_ps = cls('rand_port')
        dates = CALENDAR.td_within(20241201 , 20241207)
        for date in dates: 
            rand_ps.append(Port.rand_port(date) , override = True)
        return rand_ps

    def weight_block(self):
        """generate the weight of the portfolio as a DataBlock"""
        if not self.weight_block_completed:
            df = pd.concat([pf.df_with_date for pf in self.ports.values()] , axis = 0)
            self.weight = DataBlock.from_dataframe(df.set_index(['secid' , 'date']))
            self.weight_block_completed = True
        return self.weight
        
    def append(self , port : Port , override = False , ignore_name = False):
        """append a port to the portfolio"""
        assert ignore_name or port.name in ['none' , 'empty'] or self.name == port.name , (self.name , port.name)
        assert override or (port.date not in self.ports) , f'port {port.name} {port.date} already exists , portfolio dates: {self.port_date}'
        if port.is_emtpy(): 
            return
        self.ports[port.date] = port
        self.weight_block_completed = False
        self._last_port = port
        
    def available_dates(self): 
        """return the available dates of the portfolio"""
        return self.port_date

    def latest_avail_date(self , date : int = 99991231):
        """return the latest available date of the portfolio before the given date"""
        available_dates = self.available_dates()
        if date in available_dates: 
            return date
        tar_dates = available_dates[available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1
    def has(self , date : int):
        """return True if the portfolio has the given date"""
        return date in self.ports
    def get(self , date : int , latest = False): 
        """return the port for the given date"""
        port = self.ports.get(date , None)
        if port is None:
            use_date = self.latest_avail_date(date) if latest else date
            port = self.ports.get(use_date , None)
        if port is None: 
            return Port.none_port(date , self.name)
        else:
            return port.evolve_to_date(date)
    @classmethod
    def get_object_name(cls , obj : str | Any | None) -> str:
        """class method to get the name of the object (str , Portfolio , None)"""
        if obj is None: 
            return 'none'
        elif isinstance(obj , cls): 
            return obj.name
        elif isinstance(obj , str): 
            return obj
        else: 
            raise TypeError(obj)

    @classmethod
    def from_dataframe(cls , df : pd.DataFrame , name : str | Any = None):
        """
        class method to create a portfolio from a dataframe
        df should have columns: name , date , secid , weight , can have value column
        """
        if df.empty: 
            return cls(name)
        df = df.reset_index()
        if 'weight' not in df.columns:
            df['weight'] = 1 / len(df)
        assert all(col in df.columns for col in ['name' , 'date' , 'secid' , 'weight']) , \
            f'expect columns: name , date , secid , weight , got {df.columns.tolist()}'
        if 'value' not in df.columns: 
            df['value'] = 1
        assert df['name'].str.lower().nunique() == df['name'].nunique() , \
            f'duplicate names in dataframe considering case: {df["name"].unique()}'
        df['name'] = df['name'].str.lower()

        if name is None:
            assert df['name'].nunique() == 1 , f'all ports must have the same name , got multiple names: {df["name"].unique()}'
            name = df['name'].iloc[0]
        else:
            df = df.query('name == @name.lower()')
        
        portfolio = cls(name)
        for date , subdf in df.groupby('date'):
            portfolio.append(Port(subdf.filter(items=['secid' , 'weight']) , date , name , subdf['value'].iloc[0]))
        return portfolio
    
    def to_dataframe(self) -> pd.DataFrame:
        """convert the portfolio to a dataframe"""
        if len(self.ports) == 0: 
            return pd.DataFrame()
        else:
            df : pd.DataFrame | Any = pd.concat([port.to_dataframe() for port in self.ports.values()] , axis = 0)
            return df

    @classmethod
    def load(cls , path : Path | str) -> 'Portfolio':
        """load a portfolio from a path (dataframe)"""
        path = Path(path)
        if path.exists():
            return cls.from_dataframe(DB.load_df(path))
        else:
            return cls(path.stem)

    def save(self , path : Path | str , overwrite = False , append = True , indent : int = 1 , vb_level : int = 2):
        """save the portfolio to a path (dataframe)"""
        if self.is_empty:
            return
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path} already exists')
        path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            prev_port = DB.load_df(path)
            if not prev_port.empty:
                prev_port = prev_port.query('date < @self.port_date.min()')
            new_port = self.to_dataframe()
            port = pd.concat([prev_port , new_port]).sort_values(by=['date' , 'secid'])
        else:
            port = self.to_dataframe()
        DB.save_df(port , path , overwrite = True , prefix = f'Portfolio' , indent = indent , vb_level = vb_level)
    
    def filter_secid(self , secid : np.ndarray | Any | None = None , exclude = False , inplace = False):
        """filter the portfolio by secid , if exclude is True, filter out the secid, otherwise filter in the secid"""
        if secid is None or self.is_empty: 
            return self
        if not inplace:
            self = self.copy()
        for port in self.ports.values():
            port.filter_secid(secid , exclude)
        return self

    def filter_dates(self , dates : np.ndarray | Any | None = None , exclude = False , inplace = False):
        """filter the portfolio by dates , if exclude is True, filter out the dates, otherwise filter in the dates"""
        if dates is None or self.is_empty: 
            return self
        if not inplace:
            self = self.copy()
        if exclude:
            self.ports = {date:port for date,port in self.ports.items() if date not in dates}
        else:
            self.ports = {date:port for date,port in self.ports.items() if date in dates}
        return self
    
    def replace(self , port : 'Port|Portfolio' , inplace = False):
        """replace the portfolio with the given port or portfolio"""
        if not inplace:
            self = self.copy()
        if isinstance(port , Portfolio):
            self.ports = self.ports | port.ports
        else:
            self.ports[port.date] = port
        return self
    
    def rename(self , new_name : str):
        """rename the portfolio"""
        self._name = new_name
        for port in self.ports.values(): 
            port.rename(new_name)
        return self

    @classmethod
    def from_ports(cls , *ports : Port , name : str | None = None):
        """create a portfolio from a list of ports"""
        assert ports or name , 'expect at least one port or a name'
        portfolio = cls(name if name else ports[0].name)
        for port in ports:
            portfolio.append(port , override = True , ignore_name = bool(name))
        return portfolio
    
    def activate_accountant(self):
        """activate the accountant for the portfolio"""
        from src.res.factor.util.agency.portfolio_accountant import PortfolioAccountant
        self.accountant = PortfolioAccountant(self)
        return self
    
    def accounting(
            self , 
            benchmark : 'Portfolio | str | Any' = None ,
            start : int = -1 , end : int = 99991231 , 
            analytic = True , attribution = True , * ,
            trade_engine : Literal['default' , 'harvest' , 'yale'] | str = 'default' , 
            daily = False , cache = False , 
            resume_path : Path | str | None = None , resume_end : int | None = None , resume_drop_last = True ,
            indent : int = 0 , vb_level : int = 1
        ):
        """account the portfolio"""
        if not hasattr(self , 'accountant'): 
            self.activate_accountant()
        self.accountant.accounting(
            benchmark , start , end , analytic , attribution ,
            trade_engine = trade_engine , daily = daily , cache = cache , 
            resume_path = resume_path , resume_end = resume_end , resume_drop_last = resume_drop_last ,
            indent = indent , vb_level = vb_level)
        return self
    
    @property
    def account(self):
        """return the account of the portfolio"""
        return self.accountant.account
    
    @property
    def cached_accounts(self):
        return self.accountant.cached_accounts
    
    def activate_conditioner(self , name : str):
        from src.res.factor.util.agency.portfolio_conditioner import BaseConditioner
        if not hasattr(self , 'conditioners'):
            self.conditioners : dict[str , BaseConditioner] = {}
        if name not in self.conditioners.keys():
            conditioner = BaseConditioner.select_conditioner(name)(self)
            self.conditioners[name] = conditioner
        return self
    
    def conditioned_portfolio(self , name : str):
        assert hasattr(self , 'conditioners') and name in self.conditioners.keys() , (name , self.conditioners.keys())
        return self.conditioners[name].conditioned_portfolio()
    
    def conditioned_pf_ret(self , name : str , plot = False):
        assert hasattr(self , 'conditioners') and name in self.conditioners.keys() , (name , self.conditioners.keys())
        return self.conditioners[name].conditioned_pf_ret(plot = plot)

