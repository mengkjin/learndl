import numpy as np

from dataclasses import dataclass
from typing import Any, Literal , Optional

@dataclass
class LinearConstraint:
    '''
    A   : (K , N) exposure matrix
    type: (K , ) str, ra(w<=ub & w>=lb)、lo(w>=lb)、up(w<=ub)、fx(w=lb&w=lb)四种。
    lb  : (K , ) float, lower bound
    ub  : (K , ) float, upper bound
    '''
    A : np.ndarray
    type : np.ndarray 
    lb : np.ndarray
    ub : np.ndarray

    def __post_init__(self):
        self.check()

    def __bool__(self): return len(self) > 0
    def __len__(self): return len(self.A)
    def is_empty(self): return len(self) == 0

    def check(self , N : Optional[int] = None):
        if N is None:
            assert self.A.ndim == 2 , self.A
            assert self.type.ndim == 1 , self.type
            assert self.lb.ndim == 1 , self.lb
            assert self.ub.ndim == 1 , self.lb
            K = len(self.A)
            assert self.type.shape == (K, ) and self.ub.shape == (K, ) and self.lb.shape == (K, )
            assert np.isin(self.type , ('ra', 'lo', 'up', 'fx')).all() , np.setdiff1d(self.type , ('ra', 'lo', 'up', 'fx'))
            assert (self.lb <= self.ub).all() , self
        else:
            assert self.A.shape[1] == N , self.A

    def concat(self , other):
        if not other: return self
        assert isinstance(other , LinearConstraint) , other
        self.A = np.concatenate([self.A , other.A])
        self.type = np.concatenate([self.type , other.type])
        self.lb = np.concatenate([self.lb , other.lb])
        self.ub = np.concatenate([self.ub , other.ub])
        return self
    
    @classmethod
    def stack(cls , lin_cons : list['LinearConstraint'] , clear_after = False):
        if len(lin_cons) <= 1: return lin_cons[0]
        new_con = cls(
            np.concatenate([con.A    for con in lin_cons]) ,
            np.concatenate([con.type for con in lin_cons]) ,
            np.concatenate([con.lb   for con in lin_cons]) ,
            np.concatenate([con.ub   for con in lin_cons]) ,
        )
        if clear_after: lin_cons.clear()
        return new_con
    
    @classmethod
    def empty(cls , N : int = 1):
        return cls(np.zeros((0,N)) , np.array([]) , np.array([]) , np.array([]))

    @classmethod
    def rand(cls , N : int , L = 1):
        return cls(
            A = np.concatenate([np.ones((1,N)) , np.random.rand(L , N)] , axis=0) ,
            type = np.concatenate([['fx'] , np.repeat(['ra'] , L)]),
            lb = np.concatenate([[1.] , np.random.rand(L)]),
            ub = np.ones(1 + L),
        )

@dataclass
class BoundConstraint:
    '''
    type: str, ra(w<=ub & w>=lb)、lo(w>=lb)、up(w<=ub)、fx(w=lb&w=lb)四种。
    lb  : float, lower bound
    ub  : float, upper bound
    '''
    type : np.ndarray
    lb : np.ndarray
    ub : np.ndarray

    def __post_init__(self):
        self.check()

    def __bool__(self): return True

    def check(self , N : Optional[int] = None):
        if N is None:
            assert self.type.ndim == 1 , self.type
            assert self.lb.ndim == 1 , self.lb
            assert self.ub.ndim == 1 , self.lb
            assert np.isin(self.type , ('ra', 'lo', 'up', 'fx')).all() , np.setdiff1d(self.type , ('ra', 'lo', 'up', 'fx'))
            assert (self.lb <= self.ub).all() , np.where(self.lb > self.ub)[0]
        else:
            assert self.type.shape == (N, ) and self.ub.shape == (N, ) and self.lb.shape == (N, )

    @classmethod
    def rand(cls , N : int):
        return cls(
            type = np.repeat(['ra'] , N) ,
            lb = np.zeros(N),
            ub = np.ones(N),
        )

@dataclass
class TurnConstraint:
    '''
    bdl : float, double side constraint
    rho: float, penalty factor in objective function
    '''
    dbl : float | Any = None
    rho : float | Any = None

    def __post_init__(self): self.check()

    def __bool__(self) -> bool: return self.dbl is not None and self.dbl > 0.0

    def check(self , N : Optional[int] = None):
        if self.rho == 0: self.rho = None
        if self.dbl == 0: self.dbl = None
        assert self.rho is None or self.rho > 0.0 , self.rho
        assert self.dbl is None or self.dbl > 0.0 , self.dbl

    def rescale(self , scaler : float):
        if self.rho is not None: self.rho = self.rho / scaler
        return self

    @classmethod
    def rand(cls , N : int): return cls(dbl = 1.5)

@dataclass
class ShortConstraint:
    '''
    pos: float, total abs short upper bound
    cost: float, penalty factor in objective function
    '''
    pos  : float | Any = None
    cost : float | Any = None

    def __post_init__(self): self.check()

    def __bool__(self) -> bool: return bool(self.pos)

    def check(self , N : Optional[int] = None):
        if self.pos == 0  : self.pos  = None
        if self.cost == 0 : self.cost = None
        assert self.cost is None or self.cost >= 0.0 , self.cost
        assert self.pos is None or self.pos >= 0.0 , self.pos

    def rescale(self , scaler : float):
        if self.cost: self.cost = self.cost / scaler
        return self

    @classmethod
    def rand(cls , N : int): return cls(pos = 0.1 , cost = 0.2)

@dataclass
class CovConstraint:
    '''
    lmbd: float, risk aversion
    te  : float, tracking error constraint

    input type 1
    F: (L , N) array, common factor exposure
    C: (L , L) array, common factor covariance
    S: (N ,)  array, specific risk (can be None)

    input type 2
    cov: (N , N) array, instrument covariance
    '''
    lmbd : float | Any = None
    te   : float | Any = None
    F    : np.ndarray | Any = None
    C    : np.ndarray | Any = None
    S    : np.ndarray | Any = None
    cov  : np.ndarray | Any = None
    cov_type : Literal['normal' , 'model'] = 'model'
    
    def __post_init__(self):
        self.check()

    def __bool__(self): return self.lmbd is not None or self.te is not None

    def check(self , N : Optional[int] = None):
        if self.lmbd == 0: self.lmbd = None
        if self.te   == 0: self.te   = None

        assert self.lmbd is None or self.lmbd > 0 , self.lmbd
        assert self.te   is None or self.te > 0   , self.te
        if not self: return
        if N is None:
            if self.cov_type == 'model':
                assert self.F is None or self.F.ndim == 2 and self.C.ndim == 2 , (self.F , self.C)
                assert self.C is None or self.C.shape == (self.F.shape[0], self.F.shape[0]) , (self.F.shape , self.C.shape)
                assert self.S is None or (self.S.ndim == 1 and (self.S >= 0.0).all()) , self.S
        else:
            if self.cov_type == 'model':
                assert self.F is None or self.F.shape[-1] == N , self.F.shape
                assert self.S is None or len(self.S) == N , self.S.shape
            else:
                assert self.cov is None or self.cov.shape == (N, N)

    def rescale(self , scaler : float = 1.):
        if self.lmbd is not None: self.lmbd = self.lmbd / scaler
        return self

    def variance(self , w : np.ndarray | Any):
        if not self: np.array([0.])
        if self.cov_type == 'model':
            quad_term = w.T.dot(self.F.T).dot(self.C).dot(self.F).dot(w) 
            if self.S is not None: quad_term += (w * self.S).dot(w)
        else:
            quad_term = w.T.dot(self.cov).dot(w)
        return quad_term
    
    def model_to_normal(self):
        assert self.cov_type == 'model'
        self.cov = self.F.T.dot(self.C).dot(self.F)
        if self.S is not None: 
            ijs = np.arange(len(self.S))
            diag = self.cov[ijs,ijs] + self.S
            self.cov[ijs,ijs] = diag
        self.cov_type = 'normal'

    @classmethod
    def rand(cls , N : int , cov_type : Literal['normal' , 'model'] = 'model'):
        if cov_type == 'normal':
            v = np.random.randn(N , 10)
            cov = v @ v.T + np.eye(N) * 1e-6
            return cls(cov = cov , te = 1. , cov_type = cov_type)
        else:
            L : int = max(2 , N // 5)
            F = np.random.randn(L , N)
            v = np.random.randn(L , N)
            C = v @ v.T + np.eye(L) * 1e-6
            v = np.random.randn(N , 2 * N)
            S = v.std(axis = 1)
            return cls(F = F , C = C , S = S , te = 1. , cov_type = cov_type)
