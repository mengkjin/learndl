import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Any , ClassVar , Literal , Optional

from .....basic import CONF
STOCK_LB , STOCK_UB = CONF.SYMBOL_STOCK_LB , CONF.SYMBOL_STOCK_UB

@dataclass
class StockBound:
    lb : np.ndarray | Any = None
    ub : np.ndarray | Any = None

    def __bool__(self):
        return self.lb is not None or self.ub is not None
    
    def __len__(self):
        if isinstance(self.lb , np.ndarray):
            return len(self.lb)
        elif isinstance(self.ub , np.ndarray):
            return len(self.ub)
        else:
            return None

    def __mul__(self , other):
        return self.copy().intersect(other)
    
    def __add__(self , other):
        return self.copy().union(other)
        
    def intersect(self , other : Any):
        assert isinstance(other , StockBound) , other
        if other.lb is not None: self.update_lb(other.lb , type = 'intersect')
        if other.ub is not None: self.update_ub(other.ub , type = 'intersect')
        return self
    
    def union(self , other : Any):
        assert isinstance(other , StockBound) , other
        if other.lb is not None: self.update_lb(other.lb , type = 'union')
        if other.ub is not None: self.update_ub(other.ub , type = 'union')
        return self

    def copy(self): return deepcopy(self)

    def check(self):
        diff = self.ub - self.lb >= 0
        assert diff if isinstance(diff , bool) else diff.all() , diff
        return self

    def update_ub(self , new_val : np.ndarray | float , idx : np.ndarray | Any = None , type = 'intersect'):
        if self.ub is None: self.ub = STOCK_UB
        update_func = np.minimum if type == 'intersect' else np.maximum
        if idx is None: 
            self.ub = update_func(self.ub , new_val)
        else:
            assert idx.dtype == bool
            if isinstance(new_val , np.ndarray): new_val = new_val[idx]
            if not isinstance(self.ub , np.ndarray): self.ub = np.full(len(idx) , self.ub)
            self.ub[idx] = update_func(self.ub[idx] , new_val)
        if self.lb is not None: self.lb = np.minimum(self.ub , self.lb)

    def update_lb(self , new_val : np.ndarray | float , idx : np.ndarray | Any = None , type = 'intersect'):
        if self.lb is None: self.lb = STOCK_LB
        update_func = np.maximum if type == 'intersect' else np.minimum
        if idx is None: 
            self.lb = update_func(self.lb , new_val)
        else:
            assert idx.dtype == bool
            if isinstance(new_val , np.ndarray): new_val = new_val[idx]
            if not isinstance(self.lb , np.ndarray): self.lb = np.full(len(idx) , self.lb)
            self.lb[idx] = update_func(self.lb[idx] , new_val)
        if self.ub is not None: self.ub = np.maximum(self.ub , self.lb)

    @classmethod
    def intersect_bounds(cls , bounds : list['StockBound'] , clear_after = False) :
        assert bounds , 'must be no less than 1 bound'
        new_bound = bounds[0].copy()
        [new_bound.intersect(bnd) for bnd in bounds[1:]]
        if clear_after: bounds.clear()
        return new_bound
    
    @classmethod
    def union_bounds(cls , bounds : list['StockBound'] , clear_after = False) :
        assert bounds , 'must be no less than 1 bound'
        new_bound = bounds[0].copy()
        [new_bound.union(bnd) for bnd in bounds[1:]]
        if clear_after: bounds.clear()
        return new_bound

@dataclass
class StockPool:
    basic    : Optional[list | np.ndarray] = None
    allow    : Optional[list | np.ndarray] = None # additional basic pool , and exempted from prohibit
    ignore   : Optional[list | np.ndarray] = None # deal for opposite trades , similar as halt
    no_sell  : Optional[list | np.ndarray] = None # not for sell
    no_buy   : Optional[list | np.ndarray] = None # not for buy
    prohibit : Optional[list | np.ndarray] = None # will sell if can , will not buy
    warning  : Optional[list | np.ndarray] = None # will buy up to 0.5%
    no_ldev  : Optional[list | np.ndarray] = None # not for under bought
    no_udev  : Optional[list | np.ndarray] = None # not for over bought
    shortable: Optional[list | np.ndarray] = None # shortable stocks

    warning_ub : ClassVar[float] = 0.005

    def __bool__(self): return True

    @classmethod
    def bnd_ub(cls , secid : np.ndarray , pool : np.ndarray | list , 
               valin : np.ndarray | float = STOCK_UB , valout : np.ndarray | float = STOCK_UB):
        return StockBound(ub = np.where(np.isin(secid , pool) , valin , valout))
    
    @classmethod
    def bnd_lb(cls , secid : np.ndarray , pool : np.ndarray | list , 
               valin : np.ndarray | float = STOCK_LB , valout : np.ndarray | float = STOCK_LB):
        return StockBound(lb = np.where(np.isin(secid , pool) , valin , valout))
        
    def export(self , 
              secid : np.ndarray , # full coverage
              wb : np.ndarray | Any = None , # benchmark weight
              w0 : np.ndarray | Any = None , # original weight
        ):
        if wb is None: wb = 0
        if w0 is None: w0 = 0
        allow = self.allow if self.allow else []
        bound = StockBound(np.full(len(secid) , STOCK_LB) , np.full(len(secid) , STOCK_UB))
        
        if self.shortable is not None:
            bound.intersect(self.bnd_lb(secid , self.shortable , valout=0))

        if self.basic: 
            # not in basic pool : sell to maximum(lb , 0)
            bound.intersect(self.bnd_ub(secid , np.union1d(self.basic , allow) , valout=np.maximum(self.bound.lb,0)))

        if self.no_ldev:
            bound.intersect(self.bnd_lb(secid , self.no_ldev , wb))

        if self.no_udev:
            bound.intersect(self.bnd_ub(secid , self.no_udev , wb))

        if self.warning:
            bound.intersect(self.bnd_ub(secid , self.warning , self.warning_ub))

        if self.ignore:
            bound.intersect(self.bnd_lb(secid , self.ignore , w0))
            bound.intersect(self.bnd_ub(secid , self.ignore , w0))

        if self.no_sell:
            bound.intersect(self.bnd_lb(secid , self.no_sell , w0))

        if self.no_buy:
            bound.intersect(self.bnd_ub(secid , self.no_buy , w0))

        if self.prohibit is not None: 
            bound.intersect(self.bnd_ub(secid , np.setdiff1d(self.prohibit , allow) , 0.))

        self.bound = bound.check()
        return self.bound
    
@dataclass
class IndustryPool:
    no_sell  : Optional[list | np.ndarray] = None # not for sell
    no_buy   : Optional[list | np.ndarray] = None # not for buy
    no_net_sell : Optional[list | np.ndarray] = None # not for net sell
    no_net_buy  : Optional[list | np.ndarray] = None # not for net bought

    def __bool__(self): 
        return (self.no_sell is not None or 
                self.no_buy is not None or 
                self.no_net_sell is not None or 
                self.no_net_buy is not None)
    
    def export(self , w0 : np.ndarray | Any = None , industry : Optional[np.ndarray] = None):
        if industry is None: return StockBound()
        if w0 is None: w0 = 0
        
        lb = None if self.no_sell is None else np.where(np.isin(industry , self.no_sell) , w0 , STOCK_LB)
        ub = None if self.no_buy  is None else np.where(np.isin(industry , self.no_buy)  , w0 , STOCK_UB)

        return StockBound(lb , ub)

@dataclass
class GeneralBound:
    '''one of abs , rel , por bound'''
    key : str | Literal[f'abs' , 'rel' , 'por']
    lb : Optional[float | np.ndarray] = None
    ub : Optional[float | np.ndarray] = None

    def __post_init__(self): assert self.key in ['abs' , 'rel' , 'por'] , self.key
    def __bool__(self): return self.lb is not None or self.ub is not None
    
    def export(self , wb : np.ndarray | Any = None):
        assert wb is None or isinstance(wb , np.ndarray) , f'Only stock weight can be export , risk style/indus cannot'
        if self.key == 'abs': 
            lb , ub = self.lb , self.ub
        else:
            if wb is None:
                lb , ub = None , None
            elif self.key == 'rel' : 
                lb = None if self.lb is None else self.lb + wb
                ub = None if self.ub is None else self.ub + wb
            else: 
                lb = None if self.lb is None else self.lb * wb
                ub = None if self.ub is None else self.ub * wb
        return StockBound(lb , ub)
    
    def export_lin(self , A : np.ndarray , wb : np.ndarray | Any = None , others : list['GeneralBound'] = []) -> tuple:
        #assert not isinstance(self.lb , np.ndarray) , self.lb
        #assert not isinstance(self.ub , np.ndarray) , self.ub
        if A.ndim == 1: A = A.reshape(1,-1)
        if not self or (wb is None and self.key != 'abs'): 
            rslt = [A[:0] , np.array([]) , np.array([]) , np.array([])]
        else:
            lin_type = self.lin_con_type(self.lb , self.ub)
            # new_b = a + b * old_b
            if self.key == 'abs':   a , b = 0. , 1.
            elif self.key == 'rel': a , b = A.dot(wb) , 1.
            elif self.key == 'por': a , b = 0. , A.dot(wb)
            
            lb = STOCK_LB if self.lb is None else a + b * self.lb
            ub = STOCK_UB if self.ub is None else a + b * self.ub
            if not isinstance(lb , np.ndarray): lb = np.array([lb])
            if not isinstance(ub , np.ndarray): ub = np.array([ub])

            rslt = [A , lin_type , lb , ub]

        # assuming same A and wb (e.g. same indus and style) , merge
        for other_bnd in others:
            _rslt = other_bnd.export_lin(A , wb)
            rslt[1:] = self.lin_con_merge(rslt[1:] , _rslt[1:])

        return tuple(rslt)

    def lin_con_type(self , lb , ub):
        if lb is None: t = 'up'
        elif ub is None: t = 'lo'
        elif lb == self.ub: t = 'fx'
        else: 
            assert lb < ub , (lb , ub)
            t = 'ra'
        return np.array([t])
    
    def lin_con_merge(self , tlu_0 , tlu_1):
        lin_type_0 , lb_0 , ub_0 = tlu_0
        lin_type_1 , lb_1 , ub_1 = tlu_1

        if len(lin_type_0) == 0:
            return tlu_1
        elif len(lin_type_1) == 0:
            return tlu_0
        else:
            lb , ub = np.maximum(lb_0 , lb_1) , np.maximum(ub_0 , ub_1)
            if lin_type_0 == 'fx' or lin_type_1 == 'fx': lin_type = 'fx'
            elif lin_type_0 == 'lo' and lin_type_1 == 'lo': lin_type = 'lo'
            elif lin_type_0 == 'up' and lin_type_1 == 'up': lin_type = 'up'
            else: lin_type = 'ra'
            return np.array([lin_type]) , lb , ub
    
@dataclass
class ValidRange:
    '''one of abs , rel , por bound'''
    key : str
    lb : Optional[float] = None
    ub : Optional[float] = None

    def __post_init__(self): assert self.key in ['abs' , 'pct'] , self.key
    def __bool__(self): return self.lb is not None or self.ub is not None

    def export(self , value : np.ndarray):
        if not self: return StockBound()
        v_min , v_max= -np.inf , np.inf
        if self.lb is not None:
            v_min = self.lb if self.key == 'abs' else np.quantile(value , self.lb)

        if self.ub is not None:
            v_max = self.ub if self.key == 'abs' else np.quantile(value , self.ub)

        return StockBound(ub = (value < v_max) * (value > v_min) * STOCK_UB)

