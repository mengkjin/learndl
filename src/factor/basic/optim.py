from dataclasses import dataclass
from .var import EPS_ACCURACY
@dataclass
class SolveCond:
    '''
    additional condition for solver
    turn : if consider turnover constrains
    qobj : if consider quad objective for risk aversion
    qcon : if consider quad constraint for tracking error
    short : if consider possible short position
    '''
    turn : bool = True
    qobj : bool = True
    qcon : bool = True
    short : bool = True

@dataclass
class SolveVars:
    '''
    record number of vars or start position of vars in solver
    N : weight variable to solve
    T : turnover variable , 0 or N , must be positive
    S : shortsell variable , 0 or N , must be positive
    L : risk model factor variable , 0 or len(cov_con.F) , equals portfolio risk factor exposure
    Q : risk model quad constraint variable , 0 or 2 , equals portfolio common risk and spec risk
    '''
    N : int         # normal variable
    T : int = 0     # turnover variable
    S : int = 0     # shortsell variable
    L : int = 0     # model factor
    Q : int = 0     # quadprog factor

    def start_of(self):
        start_of   = SolveVars(0)
        start_of.T = start_of.N + self.N
        start_of.S = start_of.T + self.T
        start_of.L = start_of.S + self.S
        start_of.Q = start_of.L + self.L
        return start_of
    
class Utility:
    '''
    compute final utility of optimization
    '''
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
    '''
    record custom optimization accuracy
    '''
    def __init__(self , **kwargs) -> None:
        '''input any numerical component of accuracy function'''
        self.component = {}
        self.add(**kwargs)

    def cond_expr(self , v): return ('(√)' if v >= -EPS_ACCURACY else '(X)') + str(v)
    def __bool__(self): return all([v >= -EPS_ACCURACY for v in self.component.values()]) if self.component else False
    def __call__(self, **kwargs): return self.add(**kwargs)
    def __repr__(self): 
        return (',\n' + ' ' * 10).join([
            f'{self.__class__.__name__}(Is Accurate=[{bool(self)}] ,' ,
            *[f'{k}={self.cond_expr(v)}' for k,v in self.component.items()]
        ])
    def __mul__(self , other):
        for key , val in self.component.items(): self.component[key] = val * other
        return self
    def add(self , **kwargs): self.component.update({k:v for k,v in kwargs.items() if v is not None})

    @property
    def accurate(self):
        l = [v >= -1e-6 for v in self.component.values() if v is not None]
        return all(l) if l else False