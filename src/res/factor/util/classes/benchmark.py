"""
Benchmark class for the project
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Any , Iterable

from src.proj import DB , Const , Base , Dates
from src.data import DataBlock , DATAVENDOR

from .portfolio import Port , Portfolio

__all__ = ['Benchmark']

class Benchmark(Portfolio):
    """
    orthodox benchmark in AVAIL_BENCHMARK : csi300 , csi500 , csi800 , csi1000
    """
    _instance_dict = {}
    
    AVAILABLES = Const.Factor.BENCH.availables
    DEFAULTS   = Const.Factor.BENCH.defaults
    TESTS      = Const.Factor.BENCH.tests
    CATEGORIES = Const.Factor.BENCH.categories
    NONE       = Const.Factor.BENCH.none
    
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
        if getattr(self , 'ports' , None): 
            return # avoid double initialization
        super().__init__(name)
        if name in self.NONE:
            self.benchmark_available_dates = Dates()
        else:
            self.benchmark_available_dates = DB.dates('benchmark_ts' , self.name , use_alt = True)
        self.benchmark_attempted_dates : list[int] = []

    def __call__(self, input : Any):
        if isinstance(input , (DataBlock , pd.DataFrame)):
            return self.factor_mask(input)
        else:
            raise TypeError(input)
        
    def __bool__(self): 
        return self.name not in self.NONE

    def available_dates(self) -> Dates: 
        return self.benchmark_available_dates

    def clear(self):
        self.ports = {}
        self.benchmark_attempted_dates.clear()
        return self

    def get(self , date : int , closest = True):
        if self.name in self.NONE: 
            return Port.none_port(date , self.name)
        port = self.ports.get(date , None)
        if port is not None: 
            return port
        if not closest: 
            return Port.none_port(date , self.name)
        if DB.path('benchmark_ts' , f'{self.name}_projected' , date , use_alt = True).exists():
            port = Port(DB.load('benchmark_ts' , f'{self.name}_projected' , date) , date , self.name)
        else:
            use_date = self.closest_avail_date(date)
            if use_date in self.ports:
                port = self.ports[use_date].evolve_to_date(date)
            elif use_date in self.available_dates():
                port = Port(DB.load('benchmark_ts' , self.name , use_date , use_alt = True) , use_date , self.name)
                if use_date != date: 
                    self.append(port)
                self.benchmark_attempted_dates.append(use_date)
                port = port.evolve_to_date(date)
                if not port.emtpy:
                    DB.save(port.to_dataframe() , 'benchmark_ts' , f'{self.name}_projected' , date , vb_level = 'max')
            else:
                port = Port.none_port(date , self.name)
        assert port.date == date , (port.date , date)
        self.append(port)
        return port
    
    def get_dates(self , dates : Base.alias.intDates):
        dates = Dates(dates).diff(self.benchmark_attempted_dates)
        for d in dates: 
            self.get(d , closest = True)

    def sec_num(self , date : np.ndarray | list):
        if self:
            return np.array([self.get(d).size for d in date]) 
        else:
            return np.array([DATAVENDOR.secid(d).size for d in date])

    def universe(self , secid : np.ndarray , date : np.ndarray):
        assert self , 'No need of calculating universe for none benchmark'
        self.get_dates(date)
        weight = self.weight_block().align_secid_date(secid , date).fillna(0)
        weight.update(values = weight.values > 0 , feature = ['universe'])
        return weight
    
    def factor_mask(self , factor_val : DataBlock | pd.DataFrame):
        if not self: 
            return factor_val
        if isinstance(factor_val , DataBlock): 
            factor_val = factor_val.to_dataframe()
        factor_list = factor_val.columns.to_list()
        secid = factor_val.index.get_level_values('secid').unique().to_numpy()
        date  = factor_val.index.get_level_values('date').unique().to_numpy()
        self.get_dates(date)
        if not self.ports: 
            return factor_val
        univ  = self.universe(secid , date).to_dataframe()
        factor_val = factor_val.join(univ)
        factor_val.loc[~factor_val['universe'] , factor_list] = np.nan
        factor_val = factor_val.drop(columns=['universe'])
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
            return bm.get(model_date , closest=True)
        elif isinstance(bm , Portfolio):
            if bm.is_default and default_config and not bm.has(model_date):
                port = cls.day_port(default_config , model_date)
                bm.append(port , ignore_name = True)
            else:
                port = bm.get(model_date , closest=True)
            return port
        elif isinstance(bm , str):
            return cls(bm).get(model_date , closest=True)
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
            port.with_name(name)
            return port
        else:
            raise TypeError(bm)

    @classmethod
    def to_port(cls , benchmark : Base.alias.SingleBenchmark , date : int , evolve = True) -> Port:
        if benchmark is None:
            return Port.none_port(date)
        elif isinstance(benchmark , str):
            return cls(benchmark).get(date , closest = True)
        elif isinstance(benchmark , Portfolio):
            return benchmark.get(date , closest = True)
        elif isinstance(benchmark , Port):
            port = benchmark
            return port.evolve_to_date(date) if evolve else port.copy()
        elif hasattr(benchmark , 'benchmark') and isinstance(benchmark.benchmark , str):
            return cls(benchmark.benchmark).get(date , closest = True)
        else:
            raise TypeError(benchmark)

    @classmethod
    def to_benchmark(cls , benchmark : Base.alias.SingleBenchmark) -> Portfolio:
        if benchmark is None:
            return cls(None)
        elif isinstance(benchmark , str):
            return cls(benchmark)
        elif isinstance(benchmark , Portfolio):
            return benchmark
        elif isinstance(benchmark , Port):
            return Portfolio.from_ports(benchmark)
        elif hasattr(benchmark , 'benchmark') and isinstance(benchmark.benchmark , str):
            return cls(benchmark.benchmark)
        else:
            raise TypeError(benchmark)
        
    @classmethod
    def to_benchmarks(cls , benchmarks : Base.alias.MultipleBenchmark) -> list[Portfolio | Any]:
        if benchmarks == 'defaults': 
            return cls.defaults()
        elif not benchmarks or benchmarks == 'none': 
            return [cls(None)]
        else:
            if isinstance(benchmarks , str | Portfolio): 
                benchmarks = [benchmarks]
            benches = []
            assert isinstance(benchmarks , Iterable) , benchmarks
            for bm in benchmarks:
                if bm is None or isinstance(bm , str): 
                    benches.append(cls(bm))
                elif isinstance(bm , Portfolio): 
                    benches.append(bm)
                else: 
                    raise TypeError(bm)
        return benches

    @classmethod
    def get_benchmark_name(cls , benchmark : Base.alias.SingleBenchmark):
        if benchmark is None:
            return 'default'
        elif isinstance(benchmark , (Portfolio , Benchmark , Port)):
            return benchmark.name
        elif isinstance(benchmark , str):
            return benchmark
        elif hasattr(benchmark , 'benchmark') and isinstance(benchmark.benchmark , str):
            return benchmark.benchmark
        else:
            raise ValueError(f'Unknown benchmark type: {type(benchmark)}')
    
    @classmethod
    def defaults(cls): return [cls(bm) for bm in cls.DEFAULTS]

    @classmethod
    def as_category(cls , bm : Any):
        new_bm = np.setdiff1d(bm , cls.CATEGORIES).tolist()
        return pd.Categorical(bm , categories = cls.CATEGORIES + new_bm , ordered=True) 
    
    def accounting(self , **kwargs):
        """Benchmark cannot be accounted"""
        raise NotImplementedError('Benchmark cannot be accounted')
