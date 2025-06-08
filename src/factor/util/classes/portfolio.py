import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Any , Literal , Optional , Union

from src.data import DataBlock
from .port import Port

class Portfolio:
    '''
    portfolio realization for multiple days
    '''

    def __init__(self , name : str | Any = None) -> None:
        self.name = self.get_object_name(name)
        self.ports : dict[int,Port] = {}
        self.is_default = name is None
        self.weight_block_completed = False
        self._last_port : Port | None = None
        
    def __len__(self): return len(self.available_dates())
    def __bool__(self): return len(self) > 0
    def __repr__(self): return f'<{self.name}> : {len(self.ports)} ports'
    def __getitem__(self , date): return self.get(date)
    def __setitem__(self , date , port): 
        assert date == port.date , (date , port.date)
        self.append(port , True)
    def copy(self): return deepcopy(self)

    @property
    def port_date(self): return np.array(list(self.ports.keys()))
    @property
    def port_secid(self): return np.unique(np.concatenate([port.secid for port in self.ports.values()]))

    @classmethod
    def random(cls):  
        rand_ps = cls('rand_port')
        for _ in range(3): rand_ps.append(Port.rand_port() , override = True)
        return rand_ps

    def weight_block(self):
        if not self.weight_block_completed:
            df = pd.concat([pf.port_with_date for pf in self.ports.values()] , axis = 0)
            self.weight = DataBlock.from_dataframe(df.set_index(['secid' , 'date']))
            self.weight_block_completed = True
        return self.weight
        
    def append(self , port : Port , override = False , ignore_name = False):
        assert ignore_name or port.name in ['none' , 'empty'] or str(self.name).lower() == str(port.name).lower() , (self.name , port.name)
        assert override or (port.date not in self.ports.keys()) , (port.name , port.date)
        if port.is_emtpy(): return
        self.ports[port.date] = port
        self.weight_block_completed = False
        self._last_port = port
        
    def available_dates(self): return self.port_date

    def latest_avail_date(self , date : int = 99991231):
        available_dates = self.available_dates()
        if date in available_dates: return date
        tar_dates = available_dates[available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1
    def has(self , date : int):
        return date in self.ports.keys()
    def get(self , date : int , latest = False): 
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
        if obj is None: return 'none'
        elif isinstance(obj , cls): return obj.name
        elif isinstance(obj , str): return obj.lower()
        else: raise TypeError(obj)

    @classmethod
    def from_dataframe(cls , df : pd.DataFrame , name : str | Any = None):
        if df.empty: return cls(name)
        df = df.reset_index()
        assert all(col in df.columns for col in ['name' , 'date' , 'secid' , 'weight']) , \
            f'expect columns: name , date , secid , weight , got {df.columns.tolist()}'
        if 'value' not in df.columns: df['value'] = 1

        if name is None: 
            assert df['name'].nunique() == 1 , f'all ports must have the same name , got multiple names: {df["name"].unique()}'
            name = df['name'].iloc[0]
        else:
            df = df[df['name'] == name]
        
        portfolio = cls(name)
        for date , subdf in df.groupby('date'):
            portfolio.append(Port(subdf[['secid' , 'weight']] , date , name , subdf['value'].iloc[0]))
        return portfolio
    
    def to_dataframe(self):
        if len(self.ports) == 0: 
            return pd.DataFrame()
        else:
            return pd.concat([port.to_dataframe() for port in self.ports.values()] , axis = 0)
        
    def exclude(self , secid : np.ndarray | Any | None = None , inplace = False):
        if secid is None: return self
        if not inplace:
            self = self.copy()
        for port in self.ports.values():
            port.exclude(secid , True)
        return self
    
    def replace(self , port : Union[Port, 'Portfolio'] , inplace = False):
        if not inplace:
            self = self.copy()
        if isinstance(port , Portfolio):
            self.ports = self.ports | port.ports
        else:
            self.ports[port.date] = port
        return self

    @classmethod
    def from_ports(cls , *ports : Port , name : str | None = None):
        assert ports or name , 'expect at least one port or a name'
        portfolio = cls(name if name else ports[0].name)
        for port in ports:
            portfolio.append(port , override = True , ignore_name = bool(name))
        return portfolio
    
    def activate_accountant(self):
        from src.factor.util.agency.portfolio_accountant import PortfolioAccountant
        self.accountant = PortfolioAccountant(self)
        return self
    
    def accounting(self , 
                   benchmark : 'Portfolio | str | Any' = None ,
                   start : int = -1 , end : int = 99991231 , 
                   analytic = True , attribution = True , 
                   trade_engine : Literal['default' , 'harvest' , 'yale'] | str = 'default' , 
                   daily = False , store = False):
        if not hasattr(self , 'accountant'): self.activate_accountant()
        self.accountant.accounting(benchmark , start , end , analytic , attribution , trade_engine , daily , store)
        return self
    
    def account_with_index(self , add_index : dict[str,Any] | None = None):
        add_index = add_index or {}
        return self.accountant.account_with_index(add_index)
    
    @property
    def account(self):
        return self.accountant.account
    
    @property
    def stored_accounts(self):
        return self.accountant.stored_accounts
    
    def activate_conditioner(self , name : str):
        from src.factor.util.agency.portfolio_conditioner import BaseConditioner
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

