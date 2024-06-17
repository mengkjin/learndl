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
                 name : Optional[str] = 'port' , value : float | Any = None) -> None:
        self.exists = port is not None 
        self.port = self.EMPTY_PORT if port is None else port.groupby('secid')['weight'].sum().reset_index()
        self.date = date
        self.name = name
        self.value = value if value else 1e7
        self.sort()

    def __bool__(self): return self.name is not None
    def __repr__(self): 
        return '\n'.join([f'Portfolio <date={self.date}> <name={self.name}> <value={self.value}>: ', str(self.port)])
    def __add__(self , other): return self.merge(other)
    def __mul__(self , other):  return self.rescale(other)
    def __rmul__(self , other):  return self.rescale(other)
    def __truediv__(self , other): return self.rescale(1 / other)
    def __sub__(self , other): return self.merge(other * -1.)

    def sort(self , by : Literal['secid' , 'weight'] = 'weight' , ascending=False):
        self.port = self.port.sort_values(by , ascending=ascending).reset_index(drop=True)
        return self

    def is_emtpy(self): return not self.exists or len(self.port) == 0

    def forward(self , to : int , inplace = False):
        assert self.date >= 0 , f'Must assign date first! (now date={self.date})'
        assert to >= self.date , f'Must to a later day! ({self.date} -> {to})'
        return self.__evole(to , inplace)

    def backward(self , to : int , inplace = False):
        assert self.date >= 0 , f'Must assign date first! (now date={self.date})'
        assert to <= self.date , f'Must to a earlier day! ({self.date} -> {to})'
        return self.__evole(to , inplace)
    
    def __evole(self , to : int , inplace = False):
        rslt = self if inplace else self.copy()
        old_date = int(DATAVENDOR.latest_td(self.date))
        new_date = int(DATAVENDOR.latest_td(to))
        if old_date == new_date: return rslt

        old_long_pos  = np.round(self.long_position , 4)
        old_short_pos = np.round(self.short_position , 4)

        q0 = load_target_file('trade' , 'day', old_date)[['secid','adjfactor','close']]
        q1 = load_target_file('trade' , 'day', new_date)[['secid','adjfactor','close']]
        
        port = rslt.port.merge(q0 , on = 'secid').merge(q1 , on = 'secid')
        port['weight'] = port['weight'] * port['close_y'] * port['adjfactor_y'] / port['close_x'] / port['adjfactor_x']

        rslt.port = port.sort_values('weight' , ascending=False)
        rslt.date = to
        rslt.rebalance(old_long_pos , old_short_pos)

        return rslt
    
    def rebalance(self , long_position : float = 1., short_position : float = 0.):
        long_pos = self.port['weight'] >= 0
        now_pos = self.position
        if long_pos.any():
            assert long_position > 0 , 'unable to rebalance to 0 or negative'
            self.port.loc[long_pos , 'weight'] *= long_position / self.port.loc[long_pos , 'weight'].sum()
        else:
            long_position = 0.
        if not long_pos.all(): 
            assert short_position > 0 , 'unable to rebalance to 0 or negative'
            self.port.loc[~long_pos , 'weight'] *= -short_position / self.port.loc[~long_pos , 'weight'].sum()
        else:
            short_position = 0.
        self.value += self.value * (now_pos - self.position)
        return self

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
    @property
    def long_position(self): return self.port[self.port['weight'] > 0]['weight'].sum()
    @property
    def short_position(self): return -self.port[self.port['weight'] < 0]['weight'].sum()
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
        if not new.is_emtpy() and not another.is_emtpy():
            combined = pd.concat([new.port, another.port], ignore_index=True).groupby('secid', as_index=False)['weight'].sum()
        elif new.is_emtpy():
            combined = another.port.copy()
        else:
            combined = new.port
        new.port = combined
        return new
    
class Portfolio:
    '''Non-Consecutive stream of some portfolio'''
    def __init__(self , name : str = 'port') -> None:
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
    def __init__(self , name : str) -> None:
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