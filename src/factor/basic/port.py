import numpy as np
import pandas as pd

from copy import deepcopy
from dataclasses import dataclass
from typing import Any , Literal , Optional

from src.data import DataBlock , GetData, BlockLoader , FrameLoader , get_target_dates , load_target_file
from src.environ import RISK_INDUS , RISK_STYLE
from .vendor import DATAVENDOR

AVAIL_BENCHMARK = ['csi300' , 'csi500' , 'csi1000']

class Port:
    EMPTY_PORT = pd.DataFrame(columns=['secid','weight']).astype({'secid':int,'weight':float})

    def __init__(self , port : Optional[pd.DataFrame] , date : int = -1 , 
                 name : Optional[str] = 'port' , value : float = 1e8) -> None:
        self.exists = port is not None 
        self.port = self.EMPTY_PORT if port is None else self.sort(port.groupby('secid')['weight'].sum().reset_index())
        self.date = date
        self.name = name
        self.value = value

    def __bool__(self): return self.name is not None
    def __repr__(self): 
        return '\n'.join([f'Portfolio <date={self.date}> <name={self.name}> <value={self.value}> : ', str(self.port)])
    def __add__(self , other): return self.merge(other)
    def __mul__(self , other):  return self.rescale(other)
    def __rmul__(self , other):  return self.rescale(other)
    def __truediv__(self , other): return self.rescale(1 / other)
    def __sub__(self , other): return self.merge(other * -1.)

    def sort(self , port : pd.DataFrame ,  by : Literal['secid' , 'weight'] = 'weight' , ascending=False):
        return port.sort_values(by , ascending=ascending).reset_index(drop=True)

    def is_emtpy(self): return not self.exists or len(self.port) == 0

    def forward(self , to : int):
        assert self.date >= 0 , f'Must assign date first! (now date={self.date})'
        assert to >= self.date , f'Must to a later day! ({self.date} -> {to})'

        self.port = self.sort(self.__port_evovle(self.port , self.date , to))
        self.date = to
        return self

    def backward(self , to : int):
        assert self.date >= 0 , f'Must assign date first! (now date={self.date})'
        assert to <= self.date , f'Must to a earlier day! ({self.date} -> {to})'

        self.port = self.sort(self.__port_evovle(self.port , self.date , to))
        self.date = to
        return self

    @staticmethod
    def __port_evovle(port : pd.DataFrame , d0 : int | Any , d1 : int | Any):
        old_date = int(DATAVENDOR.latest_td(d0))
        new_date = int(DATAVENDOR.latest_td(d1))
        if old_date == new_date: return port
        q0 = load_target_file('trade' , 'day', d0)[['secid','adjfactor','close']]
        q1 = load_target_file('trade' , 'day', d1)[['secid','adjfactor','close']]
        q = port.merge(q0 , on = 'secid').merge(q1 , on = 'secid')
        q['weight'] = q['weight'] * q['close_y'] * q['adjfactor_y'] / q['close_x'] / q['adjfactor_x']
        q['weight'] = q['weight'] / q['weight'].sum(skipna=True)
        return q.loc[:,['secid','weight']]

    @classmethod
    def create(cls , secid : np.ndarray | Any , weight : np.ndarray | Any , drop0 = True, **kwargs):
        df = pd.DataFrame({'secid':secid , 'weight' : weight})
        if drop0: df = df[df['weight'] != 0]
        return cls(df , **kwargs)
    
    @classmethod
    def random(cls):
        date = np.random.randint(20200101 , 20231231)
        name = 'rand_port'
        port = pd.DataFrame({'secid' : np.random.choice(np.arange(1 , 101) , 30) , 'weight' : np.random.rand(30)})
        return cls(port,date,name).rescale()

    @classmethod
    def none_port(cls , date : int , name = 'none'): return cls(None , date , name)

    @classmethod
    def empty_port(cls , date : int , name = 'empty'): return cls(cls.EMPTY_PORT , date , name)
    
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
    @property
    def position(self): return self.port['weight'].sum()
    def copy(self): return deepcopy(self)
    def rescale(self , scale = 1. , inplace = False): 
        new = self if inplace else self.copy()
        new.port['weight'] = scale * new.port['weight'] / new.position
        return new
    def weight_align(self , secid , fillna = 0.):
        return self.port.set_index('secid').reindex(secid)['weight'].fillna(fillna).to_numpy()
    def merge(self , another , name = None , inplace = False):
        assert isinstance(another , Port) , another
        new = self if inplace else self.copy()
        if name is not None: new.name = name
        combined = pd.concat([new.port, another.port], ignore_index=True)  
        new.port = combined.groupby('secid', as_index=False)['weight'].sum()
        return new
    
class Portfolio:
    '''Non-Consecutive stream of some portfolio'''
    def __init__(self , name : Optional[str]) -> None:
        self.name = name
        self.ports : dict[int,Port] = {}
        self.weight_block_completed = False
    def __bool__(self): return self.name is not None
    def __repr__(self): return f'<{self.name}> : {len(self.ports)} ports'

    @property
    def port_date(self): return np.array(list(self.ports.keys()))
    @property
    def port_secid(self): return np.unique(np.concatenate([port.secid for port in self.ports.values()]))

    @classmethod
    def random(cls):  
        rand_ps = cls('rand_port')
        for _ in range(3): rand_ps.append(Port.random() , override = True)
        return rand_ps

    def weight_block(self):
        if not self.weight_block_completed:
            df = pd.concat([pf.port_with_date for pf in self.ports.values()] , axis = 0)
            self.weight = DataBlock.from_dataframe(df.set_index(['secid' , 'date']))
            self.weight_block_completed = True
        return self.weight
        
    def append(self , port : Port , override = False):
        assert self.name == port.name , (self.name , port.name)
        assert override or (port.date not in self.ports.keys()) , port.date
        self.ports[port.date] = port
        self.weight_block_completed = False

    def available_dates(self): return self.port_date

    def latest_avail_date(self , date : int = 99991231):
        available_dates = self.available_dates()
        if date in available_dates: return date
        tar_dates = available_dates[available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1

    def get(self , date : int , latest = False): 
        use_date = self.latest_avail_date(date) if latest else date
        port = self.ports.get(use_date , None)
        if port is None: port = Port.none_port(date)
        return port
    
class Benchmark(Portfolio):
    def __init__(self , name : Optional[str] = None) -> None:
        assert name is None or name in AVAIL_BENCHMARK , name
        super().__init__(name)
        self.benchmark_available_dates = get_target_dates('benchmark' , self.name)
        self.benchmark_attempted_dates = []

    def __call__(self, input : Any):
        if isinstance(input , (DataBlock , pd.DataFrame)):
            return self.factor_mask(input)
        else:
            raise TypeError(input)

    def available_dates(self): return self.benchmark_available_dates

    def get(self , date : int , latest = False):
        use_date = self.latest_avail_date(date) if latest else date
        port = self.ports.get(use_date , None)

        if port is None:
            if use_date in self.available_dates():
                port = Port(load_target_file('benchmark' , self.name , use_date) , date , self.name)
                self.append(port)
            else:
                port = Port.none_port(date)
            self.benchmark_attempted_dates.append(date)
        return port

    def universe(self , secid : np.ndarray , date : np.ndarray):
        assert self , 'No need of calculating universe for none benchmark'
        for d in np.setdiff1d(date , self.benchmark_attempted_dates): self.get(d , latest = True)
        weight = self.weight_block().copy().align_secid_date(secid , date).as_tensor()
        weight.values = weight.values.nan_to_num(0) > 0
        weight.feature = ['universe']

        return weight
    
    def factor_mask(self , factor_val : DataBlock | pd.DataFrame):
        if not self: return factor_val
        if isinstance(factor_val , DataBlock): factor_val = factor_val.to_dataframe()
        factor_list = factor_val.columns.to_list()
        secid = factor_val.index.get_level_values('secid').unique().values
        date  = factor_val.index.get_level_values('date').unique().values
        univ  = self.universe(secid , date).to_dataframe()
        factor_val = factor_val.join(univ)
        factor_val.loc[~factor_val['universe'] , factor_list] = np.nan
        del factor_val['universe']
        return factor_val.dropna(how = 'all')
    
    @classmethod
    def day_port(cls , bm : Port|Portfolio|str|dict|Any , model_date : int) -> Port:
        if isinstance(bm , Port):
            return bm
        elif isinstance(bm , Portfolio):
            return bm.get(model_date , latest=True)
        elif isinstance(bm , str):
            return BENCHMARKS[bm].get(model_date , latest=True)
        elif isinstance(bm , dict):
            name = str({(k if isinstance(k,(Portfolio,Port)) else str(k)):v for k,v in bm.items()})
            for i , (key , value) in enumerate(bm.items()):
                if isinstance(key,Portfolio):
                    sub_bm = key.get(model_date)
                elif isinstance(key,Port):
                    sub_bm = key
                else:
                    sub_bm = cls.day_port(key , model_date)
                port = sub_bm * value if i == 0 else port + sub_bm * value
            assert isinstance(port , Port) , port
            port.name = name
            return port
        elif bm is None:
            return Port.empty_port(model_date)
        else:
            raise TypeError(bm)

BENCHMARKS = {
    'csi300' : Benchmark('csi300') ,
    'csi500' : Benchmark('csi500') ,
    'csi1000': Benchmark('csi1000') ,
}