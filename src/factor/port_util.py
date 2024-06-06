import torch
import numpy as np
import pandas as pd

from typing import Literal , Optional

from src.data import DataBlock
from src.data.fetcher import get_target_dates , load_target_file
from src.func import match_values


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
    def __init__(self , portfolios : list[Portfolio] = []) -> None:
        self.ports = {f'{port.name}.{port.date}':port for port in portfolios}
        self.create_weight_block()

    def __repr__(self): return repr(self.ports)

    @classmethod
    def random(cls):  return cls([Portfolio.random() for _ in range(3)])

    def create_weight_block(self):
        secid = np.unique(np.concatenate([port.secid for port in self.ports.values()])) if self.ports else np.array([]).astype(int)
        date  = np.array([port.date for port in self.ports.values()]).astype(int)
        values = np.zeros((len(secid) , len(date) , 1 , 1))
        for i , port in enumerate(self.ports.values()):
            values[match_values(port.secid , secid),i,0,0] = port.weight
        self.weight = DataBlock(values , secid , date , ['weight']).align_date(np.sort(date))

    def stack(self):
        if self.ports:
            return pd.concat([pf.port_with_date for pf in self.ports.values()] , axis = 0)
        else:
            return pd.DataFrame(columns=['date','secid','weight']).astype({'date':int,'secid':int,'weight':float})
        
    def append(self , port : Portfolio):
        key = f'{port.name}.{port.date}'
        assert key not in self.ports.keys() , key
        self.ports[key] = port
        new_secid = np.union1d(self.weight.secid , port.secid)
        new_date  = np.union1d(self.weight.date  , port.date)
        self.weight = self.weight.align_secid_date(new_secid , new_date)
        i = match_values(port.secid , self.weight.secid)
        j = np.where(new_date == port.date)[0][0]
        self.weight.values[i,j,0,0] = port.weight[:]

    def get(self , name : str , date : int):
        key = f'{name}.{date}'
        return self.ports.get(key , None)

class Benchmark:
    def __init__(self , name : Optional[Literal['csi300' , 'csi500' , 'csi1000']] = None) -> None:
        self.name = name
        if self.name is None:
            self.available_dates = None
        else:
            self.available_dates = get_target_dates('benchmark' , self.name)
        self.ports = PortfolioStream()

    def latest_avail_date(self , date : int = 99991231):
        if self.available_dates is None: return None
        if date in self.available_dates: return date
        tar_dates = self.available_dates[self.available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1

    def get(self , date : int , latest = False):
        if self.name is None: return Portfolio(None , date , 'none')
        if (bm := self.ports.get(self.name , date)) is not None: return bm
        bm = load_target_file('benchmark' , self.name , date)
        if bm is None and latest: bm = load_target_file('benchmark' , self.name , self.latest_avail_date(date))
        port = Portfolio(bm , date , self.name)
        self.ports.append(port)
        return port

    def universe(self , secid : np.ndarray , date : np.ndarray):
        for i , d in enumerate(date): self.get(d , latest = i==0)
        weight = self.ports.weight.copy().align_secid_date(secid , date).as_tensor()
        weight.values = weight.values.nan_to_num(0) > 0
        weight.feature = ['universe']
        return weight