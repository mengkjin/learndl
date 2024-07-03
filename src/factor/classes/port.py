import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Any , Literal , Optional

from ...basic.factor import AVAIL_BENCHMARKS , EPS_WEIGHT
from ...data import DataBlock , get_target_dates , load_target_file
from ...data.vendor import DATAVENDOR

class Port:
    '''portfolio realization of one day'''
    EMPTY_PORT = pd.DataFrame(columns=['secid','weight']).astype({'secid':int,'weight':float})

    def __init__(self , port : Optional[pd.DataFrame] , date : int = -1 , 
                 name : str = 'default' , value : float | Any = None) -> None:
        self.exists = port is not None 
        self.port = self.EMPTY_PORT if port is None else port.groupby('secid')['weight'].sum().reset_index()
        self.date = date
        self.name = name
        self.value = value if value else 1e7
        self.sort()

    def __bool__(self): return self.exists
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

    def forward(self , n : int = 1 , inplace = True):
        if n == 0: return self if inplace else self.copy()
        assert self.date >= 0 , f'Must assign date first! (now date={self.date})'
        assert n > 0 , f'n must be non-negative! ({n})'
        return self.evolve_to_date(DATAVENDOR.td_offset(self.date , n) , inplace)

    def backward(self , n : int = -1 , inplace = True):
        if n == 0: return self if inplace else self.copy()
        assert self.date >= 0 , f'Must assign date first! (now date={self.date})'
        assert n < 0 , f'n must be non-positive! ({n})'
        return self.evolve_to_date(DATAVENDOR.td_offset(self.date , n) , inplace)
    
    def evolve_to_date(self , date : int | Any , inplace = True , rebalance = False):
        rslt = self if inplace else self.copy()
        if date == rslt.date: return rslt

        old_date = int(DATAVENDOR.latest_td(self.date))
        new_date = int(DATAVENDOR.latest_td(date))
        if old_date == new_date: 
            rslt.date = date
            return rslt

        old_pos   = rslt.long_position , rslt.short_position
        old_value = rslt.value

        q = DATAVENDOR.get_quote_ret(old_date , new_date)
        assert q is not None, f'Ret Quote (at {new_date}) is does not exists'
        port = rslt.port.merge(q , on = 'secid')
        port['new_weight'] = port['weight'] * (1 + port['ret'])

        rslt.date = date
        rslt.value = old_value * (1. + port['new_weight'].sum() - port['weight'].sum())
        port['weight'] = port['new_weight'] * old_value / rslt.value
        rslt.port = port.loc[:,['secid' , 'weight']].sort_values('weight' , ascending=False)
        if rebalance: rslt.rebalance(*old_pos)
        return rslt
    
    def fut_ret(self , new_date : Optional[int] = None) -> float:
        if not self: return 0.
        old_date = int(DATAVENDOR.latest_td(self.date))
        if new_date is None: new_date = int(DATAVENDOR.td_offset(old_date , 1))

        q = DATAVENDOR.get_quote_ret(old_date , new_date)
        assert q is not None, f'Ret Quote (at {new_date}) is does not exists'
        port = self.port.merge(q , on = 'secid').fillna(0)
        return (port['weight'] * port['ret']).to_numpy().sum()
    
    def rebalance(self , long_position : float = 1., short_position : float = 0.):
        assert long_position >= 0 and short_position >= 0 , (long_position , short_position)
        L = self.port['weight'] >= 0
        S = ~L
        if L.any(): self.port.loc[L , 'weight'] *=   long_position / self.port.loc[L,'weight'].sum()
        if S.any(): self.port.loc[S , 'weight'] *= -short_position / self.port.loc[S,'weight'].sum()
        return self

    @classmethod
    def create(cls , secid : np.ndarray | Any , weight : np.ndarray | Any , drop0 = True, **kwargs):
        weight = weight * ((weight >= EPS_WEIGHT) + (weight <= -EPS_WEIGHT))
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
    def rename(self , new_name : str):
        self.name = new_name
        return self
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
    def turnover(self , another):
        if not self or self is another: return 0.
        assert isinstance(another , Port) , another
        return (self - another).port['weight'].abs().sum()
    
class Portfolio:
    '''
    portfolio realization for multiple days
    '''
    def __init__(self , name : str = 'default' , is_default = False) -> None:
        self.name = name
        self.ports : dict[int,Port] = {}
        self.is_default = is_default
        self.weight_block_completed = False
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
        for _ in range(3): rand_ps.append(Port.random() , override = True)
        return rand_ps

    def weight_block(self):
        if not self.weight_block_completed:
            df = pd.concat([pf.port_with_date for pf in self.ports.values()] , axis = 0)
            self.weight = DataBlock.from_dataframe(df.set_index(['secid' , 'date']))
            self.weight_block_completed = True
        return self.weight
        
    def append(self , port : Port , override = False , ignore_name = False):
        assert ignore_name or self.name == port.name , (self.name , port.name)
        assert override or (port.date not in self.ports.keys()) , (port.name , port.date)
        self.ports[port.date] = port
        self.weight_block_completed = False

    def available_dates(self): return self.port_date

    def latest_avail_date(self , date : int = 99991231):
        available_dates = self.available_dates()
        if date in available_dates: return date
        tar_dates = available_dates[available_dates < date]
        return max(tar_dates) if len(tar_dates) else -1
    def has(self , date : int):
        return date in self.ports.keys()
    def get(self , date : int , latest = False): 
        use_date = self.latest_avail_date(date) if latest else date
        port = self.ports.get(use_date , None)
        if port is None: 
            return Port.none_port(date)
        else:
            return port.evolve_to_date(date)
    
class Benchmark(Portfolio):
    '''
    orthodox benchmark in AVAIL_BENCHMARK : csi300 , csi500 , csi1000
    '''
    def __init__(self , name : str) -> None:
        assert name is None or name in AVAIL_BENCHMARKS , name
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
        return port.evolve_to_date(date)

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
    def day_port(cls , bm : Port|Portfolio|str|dict|Any , model_date : int , default_config = None) -> Port:
        if bm is None:
            if default_config:
                return cls.day_port(default_config , model_date)
            else:
                return Port.empty_port(model_date)
        elif isinstance(bm , Port):
            return bm
        elif isinstance(bm , Benchmark):
            return bm.get(model_date , latest=True)
        elif isinstance(bm , Portfolio):
            if bm.is_default and default_config and not bm.has(model_date):
                port = cls.day_port(default_config , model_date)
                bm.append(port , ignore_name = True)
            else:
                port = bm.get(model_date , latest=True)
            return port
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
        else:
            raise TypeError(bm)
        
    @classmethod
    def get_benchmarks(cls , benchmarks : Optional[str] | list[Optional[str]]):
        if not isinstance(benchmarks , list): benchmarks = [benchmarks]
        benches = []
        for bm in benchmarks:
            if bm is None: benches.append(bm)
            elif isinstance(bm , str): benches.append(BENCHMARKS[bm])
            elif isinstance(bm , Portfolio): benches.append(bm)
            else: raise TypeError(bm)
        return benches

BENCHMARKS = {
    'csi300' : Benchmark('csi300') ,
    'csi500' : Benchmark('csi500') ,
    'csi800' : Benchmark('csi800') ,
    'csi1000': Benchmark('csi1000') ,
}