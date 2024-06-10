import torch
import numpy as np
import pandas as pd

from typing import Any , Literal , Optional

from src.data import DataBlock
from src.data.fetcher import get_target_dates , load_target_file
from src.func import match_values

from .data_util import DATAVENDOR

class Portfolio:
    def __init__(self , port : Optional[pd.DataFrame] , date : int = -1 , name : str = 'port') -> None:
        self.exists = port is not None 
        self.port = self.empty_port if port is None else port.groupby('secid')['weight'].sum().reset_index() 
        self.date = date
        self.name = name

    def __repr__(self): return repr(self.port)

    @classmethod
    def random(cls):
        date = np.random.randint(20200101 , 20231231)
        name = 'rand_port'
        port = pd.DataFrame({'secid' : np.arange(1,101) , 'weight' : np.random.rand(100)})
        return cls(port,date,name)

    @property
    def empty_port(self):
        return pd.DataFrame(columns=['secid','weight']).astype({'secid':int,'weight':float})
    
    @property
    def port_with_date(self):
        if len(self.port):
            return pd.DataFrame({'date':self.date , **self.port})
        else:
            return pd.DataFrame(columns=['date','secid','weight']).astype({'date':int,'secid':int,'weight':float})
        
    @property
    def secid(self): return self.port['secid'].to_numpy()
    @property
    def weight(self): return self.port['weight'].to_numpy()
    
class PortfolioStream:
    '''Non-Consecutive stream of some portfolio'''
    def __init__(self , name : Optional[str]) -> None:
        self.name = name
        self.ports : dict[int,Portfolio] = {}
        self.weight_block_completed = False

    def __repr__(self): return repr(self.ports)

    @classmethod
    def random(cls):  
        rand_ps = cls('rand_port')
        for _ in range(3): rand_ps.append(Portfolio.random() , override = True)
        return rand_ps

    def weight_block(self):
        if not self.weight_block_completed:
            df = pd.concat([pf.port_with_date for pf in self.ports.values()] , axis = 0)
            self.weight = DataBlock.from_dataframe(df.set_index(['secid' , 'date']))
            self.weight_block_completed = True
        return self.weight
        
    def append(self , port : Portfolio , override = False):
        assert self.name == port.name , (self.name , port.name)
        assert override or (port.date not in self.ports.keys()) , port.date
        self.ports[port.date] = port
        self.weight_block_completed = False

    def get(self , date : int): return self.ports.get(date , None)

class Benchmark:
    def __init__(self , name : Optional[Literal['csi300' , 'csi500' , 'csi1000']] = None) -> None:
        self.name = name
        if self.name is None:
            self.available_dates = None
        else:
            self.available_dates = get_target_dates('benchmark' , self.name)
        self.ports = PortfolioStream(self.name)

    def __bool__(self): return self.name is not None

    def __call__(self, input : Any):
        if isinstance(input , (DataBlock , pd.DataFrame)):
            return self.factor_mask(input)
        else:
            raise TypeError(input)

    def latest_avail_date(self , date : int = 99991231):
        if self.available_dates is None: return None
        if date in self.available_dates: return date
        tar_dates = self.available_dates[self.available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1

    def get(self , date : int , latest = False):
        if self.name is None: return Portfolio(None , date , 'none')
        port = self.ports.get(date)
        if port is None:
            use_date = self.latest_avail_date(date) if latest else date
            bm = load_target_file('benchmark' , self.name , use_date)
            port = Portfolio(bm , date , self.name)
            self.ports.append(port)
        return port

    def universe(self , secid : np.ndarray , date : np.ndarray):
        if self.name:
            for i , d in enumerate(date): self.get(d , latest = i==0)
            weight = self.ports.weight_block().copy().align_secid_date(secid , date).as_tensor()
            weight.values = weight.values.nan_to_num(0) > 0
            weight.feature = ['universe']
        else:
            weight = DataBlock(1 , secid , date , ['weight'])
        return weight
    
    def factor_mask(self , factor_val : DataBlock | pd.DataFrame):
        if self.name is None: return factor_val
        if isinstance(factor_val , DataBlock): factor_val = factor_val.to_dataframe()
        factor_list = factor_val.columns.to_list()
        secid = factor_val.index.get_level_values('secid').unique().values
        date  = factor_val.index.get_level_values('date').unique().values
        univ  = self.universe(secid , date).to_dataframe()
        factor_val = factor_val.join(univ)
        factor_val.loc[~factor_val['universe'] , factor_list] = np.nan
        del factor_val['universe']
        return factor_val.dropna(how = 'all')
