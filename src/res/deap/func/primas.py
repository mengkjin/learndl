import torch
from typing import Literal , Callable
from src.math import same , allna , exact
from src.math import tensor as T

class PrimTool:
    registry : dict[str, Callable] = {}
    @classmethod
    def register(cls , n_arg : Literal[1,2] = 1, **decor_kwargs):
        def decorator(func):
            func_name = func.__name__
            assert func_name not in cls.registry , f'{func_name} already registered'
            def wrapper(*args , **kwargs):
                new_func = cls.prim_legit(n_arg,**decor_kwargs)(func)
                return new_func(*args , **kwargs)
            wrapper.__name__ = func_name
            cls.registry[func_name] = wrapper
            return wrapper
        return decorator
    
    @classmethod
    def prim_legit(cls,n_arg=1,**decor_kwargs):
        assert n_arg in [1,2] , n_arg
        return cls.prim_legit_x(**decor_kwargs) if n_arg == 1 else cls.prim_legit_xy(**decor_kwargs) 
    
    @classmethod
    def prim_legit_x(cls,**decor_kwargs):
        def decorator(func):
            def wrapper(x , *args, **kwargs):
                legit = cls.input_checker(x , **decor_kwargs)
                x = func(x , *args , **kwargs) if legit else None
                if allna(x):
                    return None
                return x
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
    
    @classmethod
    def prim_legit_xy(cls,**decor_kwargs):
        def decorator(func):
            def wrapper(x , y ,*args, **kwargs):
                legit = cls.input_checker(x , y , **decor_kwargs)
                x = func(x , y , *args , **kwargs) if legit else None
                if allna(x):
                    return None
                return x
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
    
    @staticmethod
    def input_checker(*args,check_null=True,check_allna=False,check_exact=True,check_same=False,**decor_kwargs):
        if check_null and any(arg is None for arg in args):
            return False
        if check_allna and any(allna(arg) for arg in args):
            return False
        if check_exact and len(args) == 2 and exact(args[0],args[1]):
            return False
        if check_same and len(args) == 2 and same(args[0],args[1]):
            return False
        return True

@PrimTool.register(2)
def beta(x,y,*,dim=1):
    return T.beta(x,y,dim=dim)

@PrimTool.register(2)
def beta_pos(x,y,*,dim=1):
    return T.beta_pos(x,y,dim=dim)

@PrimTool.register(2)
def beta_neg(x,y,*,dim=1):
    return T.beta_neg(x,y,dim=dim)

@PrimTool.register(1)
def stddev(x , *, dim=1):
    return T.stddev(x,dim=dim)

@PrimTool.register(1)
def zscore(x , *, dim : int | None = 0 , index : int | None = None):
    return T.zscore(x,dim=dim,index=index)

@PrimTool.register(1)
def abs(x):
    return torch.abs(x)

@PrimTool.register(2)
def add(x, y):
    """x+y"""
    return T.add(x,y)

@PrimTool.register(2)
def sub(x, y):
    """x-y"""
    return T.sub(x,y)

@PrimTool.register(2)
def mul(x, y):
    """x*y"""
    return T.mul(x,y)

@PrimTool.register(2)
def div(x, y):
    """x/y"""
    return T.div(x,y)

@PrimTool.register(1)
def sigmoid(x):
    """1 / (1 + exp(-x))"""
    return T.sigmoid(x)

@PrimTool.register(2)
def rank_sub(x,y):
    """rank_pct(x) - rank_pct(y)"""
    return T.rank_sub(x,y)

@PrimTool.register(2)
def rank_add(x,y):
    """rank_pct(x) + rank_pct(y)"""
    return T.rank_add(x,y)

@PrimTool.register(2)
def rank_div(x,y):
    """rank_pct(x) / rank_pct(y)"""
    return T.rank_div(x,y)

@PrimTool.register(2)
def rank_mul(x,y):
    """rank_pct(x) * rank_pct(y)"""
    return T.rank_mul(x,y)

@PrimTool.register(1)
def log(x):
    """log(x)"""
    return T.log(x)

@PrimTool.register(1)
def sqrt(x):
    """sqrt(x)"""
    return T.sqrt(x)

@PrimTool.register(1)
def square(x):
    """x^2"""
    return T.square(x)

#@PrimaTools.prima_legit(1)
@PrimTool.register(1)
def rank_pct(x,*,dim=0):
    """rank_pct(x)"""
    return T.rank_pct(x , dim=dim)

@PrimTool.register(2)
def ts_decay_pos_dif(x, y, d):
    """rolling ending positive difference of x and y for d days"""
    return T.ts_decay_pos_dif(x , y , d)

@PrimTool.register(1)
def sign(x):
    """sign(x)"""
    return T.sign(x)

@PrimTool.register(1)
def ts_delay(x, d, * , no_alert = False):
    """delay x by d days"""
    return T.ts_delay(x , d , no_alert = no_alert)

@PrimTool.register(1)
def ts_delta(x, d , * , no_alert = False):
    """delta x by d days"""
    return T.ts_delta(x , d , no_alert = no_alert)

@PrimTool.register(1)
def scale(x, c = 1 , dim = 0):
    """scale x by c along dim"""
    return T.scale(x , c , dim=dim)

@PrimTool.register(1)
def signedpower(x, a):
    """x^a with sign(x)"""
    return T.signedpower(x , a)

@PrimTool.register(1)
def ts_zscore(x, d):
    """rolling zscore of x by d days"""
    return T.ts_zscore(x , d)

@PrimTool.register(1)
def ma(x, d):
    """rolling mean of x by d days"""
    return T.ts_mean(x , d)

@PrimTool.register(1)
def pctchg(x,d):
    """percentage change of x by d days"""
    return T.pctchg(x , d)

@PrimTool.register(1)
def ts_min(x, d):
    """rolling min of x by d days"""
    return T.ts_min(x , d)

@PrimTool.register(1)
def ts_max(x, d):
    """rolling max of x by d days"""
    return T.ts_max(x , d)

@PrimTool.register(1)
def ts_argmin(x, d):
    """rolling ending argmin of x by d days"""
    return T.ts_argmin(x , d)

@PrimTool.register(1)
def ts_argmax(x, d):
    """rolling ending argmax of x by d days"""
    return T.ts_argmax(x , d)

@PrimTool.register(1)
def ts_rank(x, d):
    """rolling ending rank of x by d days"""
    return T.ts_rank(x , d)

@PrimTool.register(1)
def ts_stddev(x, d):
    """rolling ending stddev of x by d days"""
    return T.ts_stddev(x , d)

@PrimTool.register(1)
def ts_sum(x, d):
    """rolling sum of x by d days"""
    return T.ts_sum(x , d)

@PrimTool.register(1)
def ts_product(x, d):
    """rolling product of x by d days"""
    return T.ts_product(x , d)

@PrimTool.register(1)
def ts_lin_decay(x, d):
    """rolling lin_decay of d days"""
    return T.ts_lin_decay(x , d)

@PrimTool.register(2)
def ts_corr(x , y , d):
    """rolling correlation of x and y by d days"""
    return T.ts_corr(x , y , d)

@PrimTool.register(2)
def ts_beta(x , y , d):
    """rolling beta of x and y by d days"""
    return T.ts_beta(x , y , d)

@PrimTool.register(2)
def ts_beta_pos(x , y , d):
    """rolling beta of x and y by d days, only positive x"""
    return T.ts_beta_pos(x , y , d)

@PrimTool.register(2)
def ts_beta_neg(x , y , d):
    """rolling beta of x and y by d days, only negative x"""
    return T.ts_beta_neg(x , y , d)

@PrimTool.register(2)
def ts_cov(x , y , d):
    """rolling covariance of x and y by d days"""
    return T.ts_cov(x , y , d)

@PrimTool.register(2)
def ts_rankcorr(x , y , d):
    """rolling rank correlation of x and y of d days"""
    return T.ts_rankcorr(x , y , d)

@PrimTool.register(2)
def ts_btm_y_on_x(x, y, d, n):
    """rolling bottom y on x for d days, n is the number of elements to select"""
    return T.ts_btm_y_on_x(x, y, d, n)

@PrimTool.register(2)
def ts_top_y_on_x(x, y, d, n):
    """rolling top y on x for d days, n is the number of elements to select"""
    return T.ts_top_y_on_x(x, y, d, n)

@PrimTool.register(2)
def ts_dif_y_on_x(x, y, d, n):
    """rolling difference of y on x for d days, n is the number of elements to select"""
    return T.ts_dif_y_on_x(x, y, d, n)

@PrimTool.register(1)
def ts_btm_x(x, d, n):
    """rolling bottom x for d days, n is the number of elements to select"""
    return T.ts_btm_x(x, d, n)

@PrimTool.register(1)
def ts_top_x(x, d, n):
    """rolling top x for d days, n is the number of elements to select"""
    return T.ts_top_x(x, d, n)

@PrimTool.register(1)
def ts_dif_x(x, d, n):
    """rolling difference of x for d days, n is the number of elements to select"""
    return T.ts_dif_x(x, d, n)

def all_prim_names():
    return list(PrimTool.registry.keys())
