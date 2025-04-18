import numpy as np
import pandas as pd

from typing import Any

from src.basic import PATH
from src.basic.conf import AVAIL_BENCHMARKS , DEFAULT_BENCHMARKS , CATEGORIES_BENCHMARKS
from src.data import DataBlock , DATAVENDOR

from .general import Port
from .portfolio import Portfolio
    
__all__ = ['Benchmark']

class Benchmark(Portfolio):
    '''
    orthodox benchmark in AVAIL_BENCHMARK : csi300 , csi500 , csi800 , csi1000
    '''
    _instance_dict = {}
    
    AVAILABLES = AVAIL_BENCHMARKS
    DEFAULTS   = DEFAULT_BENCHMARKS
    CATEGORIES = CATEGORIES_BENCHMARKS
    NONE       = ['none' , 'default' , 'market']
    
    def __new__(cls , name : str | Any | None , *args , **kwargs):
        name = cls.get_object_name(name)
        if name in cls._instance_dict:
            return cls._instance_dict[name]
        elif name in cls.AVAILABLES + cls.NONE:
            instance = super().__new__(cls , *args , **kwargs)
            cls._instance_dict[name] = instance
            return instance
        else:
            raise ValueError(name , cls.AVAILABLES + cls.NONE)

    def __init__(self , name : str | Any | None) -> None:
        if getattr(self , 'ports' , None): return # avoid double initialization
        super().__init__(name)
        if name in self.NONE:
            self.benchmark_available_dates = []
        else:
            self.benchmark_available_dates = PATH.db_dates('benchmark_ts' , self.name , use_alt = True)
        self.benchmark_attempted_dates = []

    def __call__(self, input : Any):
        if isinstance(input , (DataBlock , pd.DataFrame)):
            return self.factor_mask(input)
        else:
            raise TypeError(input)
        
    def __bool__(self): return self.name not in self.NONE

    def available_dates(self): return self.benchmark_available_dates

    def clear(self):
        self.ports = {}
        self.benchmark_attempted_dates = []
        return self

    def get(self , date : int , latest = True):
        if self.name in self.NONE: return Port.none_port(date , self.name)
        port = self.ports.get(date , None)
        if port is not None: return port
        if not latest: return Port.none_port(date , self.name)
        use_date = self.latest_avail_date(date)
        if use_date in self.ports:
            port = self.ports[use_date].evolve_to_date(date)
        elif use_date in self.available_dates():
            port = Port(PATH.db_load('benchmark_ts' , self.name , use_date , use_alt = True) , use_date , self.name)
            if use_date != date: self.append(port)
            self.benchmark_attempted_dates.append(use_date)
        else:
            port = Port.none_port(date , self.name)
        port = port.evolve_to_date(date)
        assert port.date == date , (port.date , date)
        self.append(port)
        return port
    
    def get_dates(self , dates : np.ndarray | list):
        for d in DATAVENDOR.CALENDAR.diffs(dates , self.benchmark_attempted_dates): 
            self.get(d , latest = True)

    def sec_num(self , date : np.ndarray | list):
        if self:
            return np.array([self.get(d).size for d in date]) 
        else:
            return np.array([DATAVENDOR.secid(d).size for d in date])

    def universe(self , secid : np.ndarray , date : np.ndarray):
        assert self , 'No need of calculating universe for none benchmark'
        self.get_dates(date)
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
        self.get_dates(date)
        if not self.ports: return factor_val
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
                return Port.none_port(model_date)
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
            return cls(bm).get(model_date , latest=True)
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
    def get_benchmarks(cls , benchmarks : Any):
        if benchmarks == 'defaults': return cls.defaults()
        elif not benchmarks or benchmarks == 'none': return [cls(None)]
        else:
            if not isinstance(benchmarks , list): benchmarks = [benchmarks]
            benches = []
            for bm in benchmarks:
                if bm is None or isinstance(bm , str): benches.append(cls(bm))
                elif isinstance(bm , Portfolio): benches.append(bm)
                else: raise TypeError(bm)
        return benches
    
    @classmethod
    def defaults(cls): return [cls(bm) for bm in cls.DEFAULTS]

    @staticmethod
    def as_category(bm : Any):
        return pd.Categorical(bm , categories = CATEGORIES_BENCHMARKS , ordered=True) 
