import time
import numpy as np

from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from typing import Any , Literal , Optional

from src.basic import CONF

from .general import Port
from .risk_model import RiskAnalytic , RISK_MODEL
from .alpha_model import AlphaModel , Amodel
from .benchmark import Benchmark
from .portfolio import Portfolio

class PortCreator(ABC):
    def __init__(self , name : str):
        self.name = name

    @abstractmethod
    def setup(self , print_info : bool = False , **kwargs): 
        return self
    
    def create(self , model_date : int , alpha_model : Optional[AlphaModel | Amodel] = None , 
               benchmark : Optional[Benchmark | Portfolio | Port] = None , init_port : Port | Any = None , 
               detail_infos = True) -> 'PortCreateResult': 
        self.model_date = model_date
        self.alpha_model = alpha_model if alpha_model is not None else Amodel.create_random(model_date)
        self.init_port = Port.none_port(model_date , self.name) if init_port is None else init_port
        self.value = self.init_port.value
        if benchmark is None:
            self.bench_port = Port.none_port(model_date)
        elif isinstance(benchmark , Port):
            self.bench_port = benchmark
        else:
            self.bench_port = benchmark.get(model_date , latest = True)
        self.detail_infos = detail_infos

        t0 = time.time()
        self.parse_input()  
        t1 = time.time()
        self.solve()
        t2 = time.time()
        self.output()
        t3 = time.time()

        self.create_result.time.update({'parse_input' : t1 - t0 , 'solve' : t2 - t1 , 'output' : t3 - t2})
        return self.create_result
    
    @abstractmethod
    def parse_input(self):
        return self

    @abstractmethod
    def solve(self):
        self.create_result : PortCreateResult
        return self

    @abstractmethod
    def output(self):
        if self.detail_infos: ...
        return self

class Utility:
    '''compute final utility of a portfolio'''
    def __init__(self , **kwargs) -> None:
        '''input any numerical component of utility function'''
        self.component = {}
        self.add(**kwargs)

    def __call__(self, **kwargs): return self.add(**kwargs)
    def __repr__(self): 
        return f'{self.__class__.__name__}(Total Utility=[{self.utility:.4f}],' + \
            ','.join([f'{k}=[{v:.4f}]' for k,v in self.component.items()]) + ')'
    def __mul__(self , other):
        for key , val in self.component.items(): self.component[key] = val * other
        return self
    
    def add(self , **kwargs): self.component.update({k:v for k,v in kwargs.items() if v is not None})

    @property
    def utility(self):
        l = [v for v in self.component.values()]
        return sum(l) if l else 0
    
class Accuarcy:
    '''record custom optimization accuracy'''
    def __init__(self , **kwargs) -> None:
        '''input any numerical component of accuracy function'''
        self.component = {}
        self.add(**kwargs)

    def cond_expr(self , v): return ('(√)' if v >= -CONF.EPS_ACCURACY else '(X)') + str(v)
    def __bool__(self): return self.accurate
    def __call__(self, **kwargs): return self.add(**kwargs)
    def __repr__(self): 
        return (',\n' + ' ' * 10).join([
            f'{self.__class__.__name__}(Is Accurate={bool(self)}' ,
            *[f'{k}={self.cond_expr(v)}' for k,v in self.component.items()]
        ])
    def __mul__(self , other):
        for key , val in self.component.items(): self.component[key] = val * other
        return self
    def add(self , **kwargs): self.component.update({k:v for k,v in kwargs.items() if v is not None})

    @property
    def accurate(self):
        l = [v >= -CONF.EPS_ACCURACY for v in self.component.values() if v is not None]
        return all(l) if l else False
    
@dataclass
class PortCreateResult:
    port        : Port
    is_success  : bool = False
    status      : Literal['optimal', 'max_iteration', 'stall'] | Any = ''
    utility     : Utility | Any = None
    accuracy    : Accuarcy | Any = None
    analytic    : RiskAnalytic | Any = None
    time        : dict[str,float] = field(default_factory=dict)

    def __post_init__(self):
        if self.utility is None: self.utility = Utility()
        if self.accuracy is None: self.accuracy = Accuarcy()

    @property
    def name(self): return self.port.name
    @property
    def date(self): return self.port.date
    @property
    def secid(self): return self.port.secid
    @property
    def w(self): return self.port.weight

    def analyze(self , bench_port : Port | None = None , init_port : Port | None = None):
        port = Port.none_port(self.date) if self.port is None else self.port
        self.analytic = RISK_MODEL.get(self.date).analyze(port , bench_port , init_port)
        return self

    def __repr__(self):
        info_list = [
            f'{self.__class__.__name__} ---------- ' ,
            self.create_information() , 
            self.create_result() , 
            self.port_information() , 
            self.risk_analytic() ,
            f'Other components include [\'w\' , \'secid\'])' ,
        ]
        return '\n'.join([s for s in info_list if s is not None])
    
    def create_information(self):
        if not self.status: return None
        return '\n'.join([
            f'Creation Information : ' ,
            f'    is_success = {self.is_success} ,' ,
            f'    status     = {self.status} ,' ,
        ])
    
    def create_result(self):
        if not self.status: return None
        return '\n'.join([
            f'Creation Result : ' ,
            f'    utility    = {self.utility}' ,
            f'    accuracy   = {self.accuracy}' ,
        ]) 
    
    def port_information(self):
        if not self.port: return None
        return str(self.port)
    
    def risk_analytic(self):
        if not self.analytic: return None
        return '\n'.join([
            f'Analytic : (Only show style , access industry/risk mannually)' ,
            self.analytic.styler('style').to_string() ,
        ])
