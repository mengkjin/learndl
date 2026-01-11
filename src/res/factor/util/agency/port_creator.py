from abc import ABC , abstractmethod
from dataclasses import dataclass , field
from datetime import datetime
from typing import Any , Literal

from ..classes import Port , Benchmark , Portfolio , AlphaModel , Amodel , RiskAnalytic , RISK_MODEL

class PortCreator(ABC):
    def __init__(self , name : str):
        self.name = name

    @abstractmethod
    def setup(self , indent : int = 2 , vb_level : int = 3 , **kwargs): 
        return self
    
    def create(self , model_date : int , alpha_model : AlphaModel | Amodel | None = None , 
               benchmark : Benchmark | Portfolio | Port | None = None , init_port : Port | Any = None , 
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

        t0 = datetime.now()
        self.parse_input()  
        t1 = datetime.now()
        self.solve()
        t2 = datetime.now()
        self.output()
        t3 = datetime.now()

        self.create_result.timecost.update({
            'parse' : (t1 - t0).total_seconds() ,
            'solve' : (t2 - t1).total_seconds() , 
            'output' : (t3 - t2).total_seconds()})
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
        if self.detail_infos: 
            self.create_result.analyze(self.bench_port , self.init_port)
        return self

class PortCreateUtility:
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
        for key , val in self.component.items(): 
            self.component[key] = val * other
        return self
    
    def add(self , **kwargs): self.component.update({k:v for k,v in kwargs.items() if v is not None})

    @property
    def utility(self):
        utilities = [v for v in self.component.values()]
        return sum(utilities) if utilities else 0
    
class PortCreateAccuracy:
    '''record custom optimization accuracy'''
    EPS = 1e-5
    EPS_DICT = {'excess_turn' : 1e-4}

    def __init__(self , **kwargs) -> None:
        '''input any numerical component of accuracy function'''
        self.component = {}
        self.add(**kwargs)

    def cond_expr(self , k : str , v): 
        return ('(√)' if self.cond_assess(k , v) else '(X)') + str(v)
    def cond_assess(self , k : str , v):
        if v is None: 
            return True
        return v >= -self.EPS_DICT.get(k , self.EPS)
    def __bool__(self): 
        return self.accurate
    def __call__(self, **kwargs): 
        return self.add(**kwargs)
    def __repr__(self): 
        return (',\n' + ' ' * 10).join([
            f'{self.__class__.__name__}(Is Accurate={bool(self)}' ,
            *[f'{k}={self.cond_expr(k , v)}' for k,v in self.component.items()]
        ])
    def __mul__(self , other):
        for key , val in self.component.items(): 
            self.component[key] = val * other
        return self
    def add(self , **kwargs): self.component.update({k:v for k,v in kwargs.items() if v is not None})

    @property
    def accurate(self):
        return all([self.cond_assess(k , v) for k , v in self.component.items()])
    
@dataclass
class PortCreateResult:
    port        : Port
    is_success  : bool = False
    status      : Literal['optimal', 'max_iteration', 'stall'] | Any = ''
    utility     : PortCreateUtility | Any = None
    accuracy    : PortCreateAccuracy | Any = None
    analytic    : RiskAnalytic | Any = None
    timecost    : dict[str,float] = field(default_factory=dict)

    def __post_init__(self):
        if self.utility is None: 
            self.utility = PortCreateUtility()
        if self.accuracy is None: 
            self.accuracy = PortCreateAccuracy()

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
        if not self.status: 
            return None
        return '\n'.join([
            f'Creation Information : ' ,
            f'    is_success = {self.is_success} ,' ,
            f'    status     = {self.status} ,' ,
        ])
    
    def create_result(self):
        if not self.status: 
            return None
        return '\n'.join([
            f'Creation Result : ' ,
            f'    utility    = {self.utility}' ,
            f'    accuracy   = {self.accuracy}' ,
        ]) 
    
    def port_information(self):
        if not self.port: 
            return None
        return str(self.port)
    
    def risk_analytic(self):
        if not self.analytic: 
            return None
        return '\n'.join([
            f'Analytic : (Only show style , access industry/risk mannually)' ,
            str(self.analytic.styler('style')) ,
        ])
