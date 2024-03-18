import torch
from torch.nn.functional import pad , one_hot
import numpy as np
from sklearn.linear_model import LinearRegression
#from numba import jit , cuda
#import pandas as pd


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#import os
#if os.name == 'posix': assert(device == torch.device('cuda'))
#print(__name__)

NaN = torch.nan

def allna(x , inf_as_na = True):
    if inf_as_na:
        return not x.isfinite().any()
    else:
        return x.isnan().all()

def exact(x , y):
    return x is y

def same(x , y):
    if type(x) != type(y):
        return False
    else:
        return x.equal(y)

class PrimaTools:
    @classmethod
    def decor(cls , n_arg = 1, roller = False , **decor_kwargs):
        def decorator(func):
            def wrapper(*args , **kwargs):
                new_func = cls.ts_roller(n_arg,**decor_kwargs)(func) if roller else func
                new_func = cls.prima_legit(n_arg,**decor_kwargs)(new_func)
                return new_func(*args , **kwargs)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
    
    @classmethod
    def prima_legit(cls,n_arg=1,**decor_kwargs):
        assert n_arg in [1,2] , n_arg
        return cls._prima_legit_1(**decor_kwargs) if n_arg == 1 else cls._prima_legit_2(**decor_kwargs) 
    
    @classmethod
    def ts_roller(cls,n_arg=1,**decor_kwargs):
        assert n_arg in [1,2] , n_arg
        return cls._roller_1(**decor_kwargs) if n_arg == 1 else cls._roller_2(**decor_kwargs)

    @classmethod
    def _prima_legit_1(cls,**decor_kwargs):
        def decorator(func):
            def wrapper(x , *args, **kwargs):
                legit = cls.legit_checker((x,),**decor_kwargs)
                if not legit: return None
                return func(x , *args , **kwargs)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
    
    @classmethod
    def _prima_legit_2(cls,**decor_kwargs):
        def decorator(func):
            def wrapper(x , y ,*args, **kwargs):
                legit = cls.legit_checker((x,y),**decor_kwargs)
                if not legit: return None
                return func(x , y , *args , **kwargs)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
    
    @classmethod
    def _roller_1(cls , nan = NaN , pinf = torch.inf , ninf = -torch.inf , **decor_kwargs):
        def decorator(func):
            def wrapper(x , d , *args, **kwargs):
                assert d <= len(x) , (d,x)
                x = x.nan_to_num(nan,pinf,ninf)
                x = func(x.unfold(0,d,1) , d , *args, **kwargs)
                return pad(x , (0,0,d-1,0) , value = NaN)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
    
    @classmethod
    def _roller_2(cls , nan = NaN , pinf = torch.inf , ninf = -torch.inf , **decor_kwargs):
        def decorator(func):
            def wrapper(x , y , d , *args, **kwargs):
                assert d <= len(x) , (d,x)
                x = x.nan_to_num(nan,pinf,ninf)
                y = y.nan_to_num(nan,pinf,ninf)
                z = func(x.unfold(0,d,1) , y.unfold(0,d,1) , d , *args, **kwargs)
                return pad(z , (0,0,d-1,0) , value = NaN)
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator
    
    @staticmethod
    def legit_checker(args,/,check_null=1,check_exact=1,check_allna=0,check_same=0,**decor_kwargs):
        for arg in args: 
            if check_null  and arg is None: return False
            if check_allna and allna(arg):  return False
        if len(args) == 2 and check_exact and exact(args[0],args[1]): return False
        if len(args) == 2 and check_same  and same(args[0],args[1]):  return False
        return True

"""
def prima_legit(check_valid = 1 , check_exact = True , check_allna = False , check_same = False):
    def decorator(func):
        def wrapper(*args , **kwargs):
            if check_valid >= 1:
                if args[0] is None: return None
                if check_allna and allna(args[0]): return None
            if check_valid >= 2:
                if args[1] is None: return None
                if check_allna and allna(args[1]): return None
                if check_exact and exact(args[0],args[1]): return None
                if check_same  and same(args[0],args[1]): return None
            v = func(*args , **kwargs)
            return v
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

def ts_roller(nan = NaN , pinf = torch.inf , ninf = -torch.inf):
    def decorator(func):
        def wrapper(x , d , *args, **kwargs):
            assert d <= len(x) , (d,x)
            x = x.nan_to_num(nan,pinf,ninf)
            x = func(x.unfold(0,d,1) , d , *args, **kwargs)
            x = pad(x , (0,0,d-1,0) , value = NaN)
            return x
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

def ts_coroller(nan = NaN , pinf = torch.inf , ninf = -torch.inf):
    def decorator(func):
        def wrapper(x , y , d , *args, **kwargs):
            assert d <= len(x) , (d,x)
            x = x.nan_to_num(nan,pinf,ninf)
            y = y.nan_to_num(nan,pinf,ninf)
            z = func(x.unfold(0,d,1) , y.unfold(0,d,1) , d , *args, **kwargs)
            z = pad(z , (0,0,d-1,0) , value = NaN)
            return z
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator
"""

#@PrimaTools.prima_legit(2,check_exact=0)
@PrimaTools.decor(2,check_exact=0)
def rankic_2d(x , y , dim = 1 , universe = None , min_coverage = 0.5):
    valid = ~y.isnan()
    if universe is not None: valid *= universe.nan_to_num(False)
    x = torch.where(valid , x , NaN)

    coverage = (~x.isnan()).sum(dim=dim)
    x = rank_pct(x)
    y = rank_pct(y)
    ic = corrwith(x , y , dim=dim)
    return ic if ic is None else torch.where(coverage < min_coverage * valid.sum(dim=dim) , NaN , ic)

#%% 中性化函数
def dummy(x , ex_last = True):
    # will take a huge amount of memory, the suggestion is to use torch.cuda.empty_cache() after neutralization
    if not isinstance(x , torch.Tensor): x = torch.Tensor(x)
    xmax = x.nan_to_num().max().int().item() 
    dummy = x.nan_to_num(xmax + 1).to(torch.int64)
    # dummy = torch.where(x.isnan() , m + 1 , x).to(torch.int64)
    dummy = one_hot(dummy).to(torch.float)[...,:xmax+1-ex_last] # slightly faster but will take a huge amount of memory
    dummy = dummy[...,dummy.sum(dim = tuple(range(x.dim()))) != 0]
    return dummy

def concat_factors(*factors , n_dims = 2 , dim = -1 , device = None):
    if len(factors) == 0: return None
    factors = list(factors)
    
    for i in range(len(factors)):
        #if factors[i] is None: factors[i] = None
        if isinstance(factors[i] , torch.Tensor):
            if factors[i].dim() == n_dims:  factors[i] = factors[i].unsqueeze(dim)
            if device is not None: factors[i] = factors[i].to(device)

    factors = [f for f in factors if f is not None]
    factors = factors[0] if len(factors) == 1 else torch.cat(factors , dim = dim)
    return factors

def betas_torch(x , y):
    try:
        b = torch.linalg.lstsq(x , y , rcond=None)[0]
    except: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
        try:    
            b = torch.linalg.inv(x.T.mm(x)).mm(x.T).mm(y)
        except:
            print('neutralization error!')
            b = torch.zeros(x.shape[-1],1).to(x)
    return b

def betas_np(x , y):
    try:
        b = np.linalg.lstsq(x , y , rcond=None)[0]
    except: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
        try:    
            b = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        except:
            print('neutralization error!')
            b = np.zeros((x.shape[-1],1))
    return b

def betas_sk(x , y):
    try:
        b = LinearRegression(fit_intercept=False).fit(x, y).coef_.T
    except: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
        print('neutralization error!')
        b = np.zeros((x.shape[-1],1))
    return b

def beta_calculator(method = 'torch'):
    assert method in ['torch' , 'np' , 'sk'] , method
    if method == 'np':
        betas_func = betas_np
    elif method == 'sk':
        betas_func = betas_sk
    else:
        betas_func = betas_torch
    return betas_func

def neutralize_xdata_2d(factors = None , groups = None):
    # not the most time efficient way, since its is very memory consuming
    if groups is None: groups  = []
    if not isinstance(groups , (list , tuple)): groups  = [groups]
    x = concat_factors(factors , n_dims = 2)
    for g in groups: 
        x = concat_factors(x , dummy(g , ex_last=True) , n_dims = 2)
    if isinstance(x , torch.Tensor):
        x.nan_to_num_(NaN , NaN , NaN)
        x = pad(x , (1,0) , value = 1.)
    return x

def neutralize_2d(y , x , dim = 1 , method = 'torch' , device = None , inplace = False):  # [tensor (TS*C), tensor (TS*C)]
    if x  is None or y is None: return y

    assert method in ['sk' , 'np' , 'torch']
    assert dim in [-1,0,1] , dim
    assert y.dim() == 2 , y.dim()
    assert x.dim() in [2,3] , x.dim()
    
    old_device = y.device
    finite_ij  = x[...,0].isfinite()
    for k in range(1, x.shape[-1]): finite_ij += x[...,k].isfinite()
    finite_ij *= y.isfinite() 
    x = x.nan_to_num(0,0,0)
    if inplace:
        y.nan_to_num_(NaN , NaN , NaN).unsqueeze_(-1)
    else:
        y = y.nan_to_num(NaN , NaN , NaN).unsqueeze(-1)
    nonzero_ik = ~(x == 0).all(dim)
    if device is not None:  x , y = x.to(device) , y.to(device)
    if dim == 0: x , y = x.permute(1,0,2) , y.permute(1,0,2)
    res = None
    if False and method == 'torch' and finite_ij.all() and nonzero_ik.all():
        # fastest, but cannot deal NaN's which is always the case, so will not enter here
        try:
            model = torch.linalg.lstsq(x , y , rcond=None)
            res = (y - x @ model[0])
        except:
            res = None
    if res is not None:
        y = res
    else: 
        betas_func = beta_calculator(method)
        if method in ['sk' , 'np']: 
            x , y , finite_ij , nonzero_ik = x.cpu().numpy() , y.cpu().numpy() , finite_ij.cpu().numpy()  , nonzero_ik.cpu().numpy() 
        for i , (y_ , x_ , j_ , k_) in enumerate(zip(y , x , finite_ij , nonzero_ik)):
            if j_.sum() < 10: continue
            betas = betas_func(x_[:,k_][j_] , y_[j_])
            y[i] -= x_[:,k_] @ betas
        if isinstance(y , np.ndarray): y = torch.Tensor(y)
    if dim == 0: y = y.permute(1,0,2)
    y = zscore_inplace(y.squeeze_(-1) , dim = dim).to(old_device)
    return y

def neutralize_1d(y , x , insample , method = 'torch' , device = None , inplace = False):  # [tensor (TS*C), tensor (TS*C)]
    if x is None or y is None: return y

    assert method in ['sk' , 'np' , 'torch']
    assert y.shape == insample.shape , (y.shape , insample.shape)
    assert y.dim() == 1 , y.dim()
    
    if x.dim() == 1: x = x.reshape(-1,1)
    old_device = y.device
    finite_i  = x[:,0].isfinite()
    for k in range(1, x.shape[-1]): finite_i += x[:,k].isfinite()
    finite_i *= y.isfinite() 
    if finite_i.sum() < 10: return y

    x = x.nan_to_num(0,0,0)
    if inplace:
        y.nan_to_num_(NaN , NaN , NaN).unsqueeze_(-1)
    else:
        y = y.nan_to_num(NaN , NaN , NaN).unsqueeze(-1)
    
    if device is not None:  x , y = x.to(device) , y.to(device)
    betas_func = beta_calculator(method)
    if method in ['sk' , 'np']: 
        x , y , finite_i = x.cpu().numpy() , y.cpu().numpy() , finite_i.cpu().numpy()

    x_ , y_ , f_ = x[insample] , y[insample] , finite_i[insample]
    k_ = ~(x_ == 0).all(0)
    betas = betas_func(x_[f_][:,k_] , y_[f_])
    y -= x[:,k_] @ betas
    if isinstance(y , np.ndarray): y = torch.Tensor(y)
    y = zscore_inplace(y.squeeze_(-1) , dim = 0).to(old_device)
    return y

#%% 相关系数函数
# 将以上函数以矩阵形式改写
#@PrimaTools.prima_legit(2,check_exact=0)
@PrimaTools.decor(2,check_exact=0)
def corrwith(x,y,dim=None):
    if same(x , y): return torch.where(x.nansum(dim) > 2 , 1 , NaN)
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  
    y_ymean = y - torch.nanmean(y, dim, keepdim=True) 
    cov  = torch.nansum(x_xmean * y_ymean, dim) 
    ssd  = x_xmean.square().nansum(dim).sqrt() * y_ymean.square().nansum(dim).sqrt()
    ssd[ssd == 0] = 1e-4
    corr = cov / ssd
    return corr

#@PrimaTools.prima_legit(2,check_exact=0)
@PrimaTools.decor(2,check_exact=0)
def covariance(x,y,dim=None):
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  # [TS, C]
    y_ymean = y - torch.nanmean(y, dim, keepdim=True)  # [TS, C]
    cov = torch.nansum(x_xmean * y_ymean, dim)  # [TS, 1]
    return cov

#@PrimaTools.prima_legit(2,check_exact=0)
@PrimaTools.decor(2,check_exact=0)
def beta(x,y,dim=None):
    if same(x , y): return torch.where(x.nansum(dim) > 2 , 1 , NaN)
    return covariance(x,y,dim) / covariance(x,x,dim)

#@PrimaTools.prima_legit(2,check_exact=0)
@PrimaTools.decor(2,check_exact=0)
def beta_pos(x,y,dim=None):
    if same(x , y): return torch.where(x.nansum(dim) > 2 , 1 , NaN)
    y = torch.where(x < 0 , NaN , y)
    x = torch.where(x < 0 , NaN , x)
    return covariance(x,y,dim) / covariance(x,x,dim)

#@PrimaTools.prima_legit(2,check_exact=0)
@PrimaTools.decor(2,check_exact=0)
def beta_neg(x,y,dim=None):
    if same(x , y): return torch.where(x.nansum(dim) > 2 , 1 , NaN)
    y = torch.where(x > 0 , NaN , y)
    x = torch.where(x > 0 , NaN , x)
    return covariance(x,y,dim) / covariance(x,x,dim)

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def stddev(x , dim = 0 , keepdim = True):
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    return torch.nansum(x_xmean ** 2, dim , keepdim = keepdim).sqrt()

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def zscore(x , dim = 0 , index = None):
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = torch.nansum(x_xmean ** 2, dim , keepdim=index is None).sqrt()
    if index: x_xmean = x_xmean.select(dim,index)
    z = x_xmean / (x_stddev + 1e-4 * x_stddev.nanmean())
    return z

@PrimaTools.decor(1)
def abs(x):
    return torch.abs(x)

@PrimaTools.decor(1)
def nanstd(x , dim = 0):
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = torch.nansum(x_xmean ** 2, dim , keepdim=False).sqrt()
    return x_stddev

@PrimaTools.decor(1)
def zscore_inplace(x , dim = 0):
    x -= torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = torch.nansum(x ** 2, dim , keepdim=True).sqrt()
    x /= (x_stddev + 1e-4 * x_stddev.nanmean())
    return x

#%% 其他函数
#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def add(x, y):
    'x+y'
    return x + y

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def sub(x, y):
    'x-y'
    return x - y

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def mul(x, y):
    'x*y'
    return x * y

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def div(x, y):
    'x/y'
    return x / y

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def add_int(x, d):
    'x+d'
    return x + d

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def sub_int1(x, d):
    'x-d'
    return x - d

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def sub_int2(x, d):
    'x-d'
    return x - d

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def mul_int(x, d):
    'x*d'
    return x * d

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def div_int1(x, d):
    'x/d'
    return x / d

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def div_int2(x, d):
    'x/d'
    return x / d

def neg(x):
    '-x'
    return x if x is None else -x

def neg_int(x):
    '-x'
    return -x

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def rank_sub(x,y):
    return rank_pct(x,1) - rank_pct(y,1)

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def rank_add(x,y):
    return rank_pct(x,1) + rank_pct(y,1)

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def rank_div(x,y):
    return rank_pct(x,1) / rank_pct(y,1)

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def rank_mul(x,y):
    return rank_pct(x,1) * rank_pct(y,1)

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def log(x):
    return x.log()

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def sqrt(x):
    return x.sqrt()

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def square(x):
    return x.square()

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def rank_pct(x,dim=1):
    assert (len(x.shape) <= 3)
    x_rank = x.argsort(dim=dim).argsort(dim=dim).to(torch.float32) + 1 # .where(~x.isnan() , NaN)
    x_rank[x.isnan()] = NaN
    x_rank = x_rank / ((~x_rank.isnan()).sum(dim=dim, keepdim=True))
    return x_rank

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def lin_decay(x , dim = 0):   #only for rolling
    'd日衰减加权平均，加权系数为 d, d-1,...,1'
    shape_ = [1] * len(x.shape)
    shape_[dim] = x.shape[dim]
    coef = torch.arange(1, x.shape[dim] + 1, 1).reshape(shape_).to(x)
    return (x * coef).nansum(dim=dim) / ((coef * (~x.isnan())).sum(dim=dim))

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def ts_decay_pos_dif(x, y, d):
    value = x - y
    value[value < 0] = 0
    return ts_lin_decay(value, d)

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def sign(x):
    return x.sign()

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def ts_delay(x, d):
    if d > x.shape[0]: return None
    if d < 0: print('Beware! future information used!')
    z = x.roll(d, dims=0)
    if d >= 0:
        z[:d,:] = NaN
    else:
        z[d:,:] = NaN
    return z

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def ts_delta(x, d):
    if d > x.shape[0]: return None
    if d < 0: print('Beware! future information used!')
    z = x - ts_delay(x, d)
    return z

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def scale(x, c = 1 , dim = 1):
    return c * x / x.abs().nansum(axis=dim, keepdim=True)

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def signedpower(x, a):
    return x.sign() * x.abs().pow(a)

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True)
def ts_zscore(x, d):
    return zscore(x , dim = -1 , index = -1)

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True)
def ma(x, d):
    return torch.nanmean(x , dim=-1)

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def pctchg(x,d):
    return (x - ts_delay(x,d)) / abs(ts_delay(x,d))

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True,nan=np.inf)
def ts_min(x, d):
    return torch.min(x , dim=-1)[0]

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True,nan=-np.inf)
def ts_max(x, d):
    return torch.max(x , dim=-1)[0]

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True,nan=np.inf)
def ts_argmin(x, d):
    return torch.argmin(x , dim=-1).to(torch.float)

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True,nan=-np.inf)
def ts_argmax(x, d):
    return torch.argmax(x , dim=-1).to(torch.float)

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True)
def ts_rank(x, d):
    return rank_pct(x,dim=-1)[...,-1]

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True,nan=0)
def ts_stddev(x, d):
    return torch.std(x,dim=-1)

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True,nan=0)
def ts_sum(x, d):
    return torch.sum(x,dim=-1)

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True,nan=1)
def ts_product(x, d):
    return torch.prod(x,dim=-1)

#@PrimaTools.prima_legit(1)
#@ts_roller()
@PrimaTools.decor(1,roller=True)
def ts_lin_decay(x, d):
    return lin_decay(x , dim=-1)

#@PrimaTools.prima_legit(2)
#@ts_coroller(0)
@PrimaTools.decor(2,roller=True,nan=0)
def ts_corr(x , y , d):
    return corrwith(x , y , dim=-1)

#@PrimaTools.prima_legit(2)
#@ts_coroller(0)
@PrimaTools.decor(2,roller=True,nan=0)
def ts_beta(x , y , d):
    return beta(x , y , dim=-1)

#@PrimaTools.prima_legit(2)
#@ts_coroller(0)
@PrimaTools.decor(2,roller=True,nan=0)
def ts_beta_pos(x , y , d):
    return beta_pos(x , y , dim=-1)

#@PrimaTools.prima_legit(2)
#@ts_coroller(0)
@PrimaTools.decor(2,roller=True,nan=0)
def ts_beta_neg(x , y , d):
    return beta_neg(x , y , dim=-1)

#@PrimaTools.prima_legit(2)
#@ts_coroller(0)
@PrimaTools.decor(2,roller=True,nan=0)
def ts_cov(x , y , d):
    return covariance(x , y , dim=-1)

#@PrimaTools.prima_legit(2)
#@ts_coroller(0)
@PrimaTools.decor(2,roller=True,nan=0)
def ts_rankcorr(x , y , d):
    return corrwith(rank_pct(x,dim=-1) , rank_pct(y,dim=-1) , dim=-1)

def rlbxy(x, y, d, n, btm, sel_posneg=False):
    assert btm in ['btm' , 'top' , 'diff'] , btm
    n = min(d, n)
    x , y = x.unfold(0,d,1) , y.unfold(0,d,1)
    groups = [None , None]
    if btm in ['btm' , 'diff']:
        groups[0] = (x <= x.kthvalue(n, dim=-1, keepdim=True)[0])
        if sel_posneg: groups[0] *= (x < 0)
    if btm in ['top' , 'diff']:
        groups[1] = (x >= x.kthvalue(d-n+1, dim=-1, keepdim=True)[0])
        if sel_posneg: groups[1] *= (x > 0)
        
    z = [torch.where(grp , y , NaN).nanmean(dim=-1) for grp in groups if grp is not None]
    z = pad(z[0] if len(z) == 1 else (z[1] - z[0]) , (0,0,d-1,0) , value = NaN)
    return z

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def ts_xbtm_yavg(x, y, d, n):
    '在过去d日上，根据x进行排序，取最小n个x的y的平均值'
    return rlbxy(x, y, d, n, btm='btm')

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def ts_xtop_yavg(x, y, d, n):
    '在过去d日上，根据x进行排序，取最大n个x的y的平均值'
    return rlbxy(x, y, d, n, btm='top')

#@PrimaTools.prima_legit(2)
@PrimaTools.decor(2)
def ts_xrng_ydif(x, y, d, n):
    '在过去d日上，根据x进行排序，取最大n个x的y的平均值与最小n个x的y的平均值的差值'
    return rlbxy(x, y, d, n, btm='diff')

def rlbx(x, d, n, btm, sel_posneg=False):
    assert btm in ['btm' , 'top' , 'diff'] , btm
    n = min(d, n)
    x = x.unfold(0,d,1)
    groups = [None , None]
    if btm in ['btm' , 'diff']:
        groups[0] = (x <= x.kthvalue(n, dim=-1, keepdim=True)[0])
        if sel_posneg: groups[0] *= (x < 0)
    if btm in ['top' , 'diff']:
        groups[1] = (x >= x.kthvalue(d-n+1, dim=-1, keepdim=True)[0])
        if sel_posneg: groups[1] *= (x > 0)
        
    z = [torch.where(grp , x , NaN).nanmean(dim=-1) for grp in groups if grp is not None]
    z = pad(z[0] if len(z) == 1 else (z[1] - z[0]) , (0,0,d-1,0) , value = NaN)
    return z

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def ts_btm_avg(x, d, n):
    '在过去d日上，根据x进行排序，取最小n个x的y的平均值'
    return rlbx(x, d, n, btm='btm')

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def ts_top_avg(x, d, n):
    '在过去d日上，根据x进行排序，取最大n个x的y的平均值'
    return rlbx(x, d, n, btm='top')

#@PrimaTools.prima_legit(1)
@PrimaTools.decor(1)
def ts_rng_dif(x, d, n):
    '在过去d日上，根据x进行排序，取最大n个x的y的平均值与最小n个x的y的平均值的差值'
    return rlbx(x, d, n, btm='diff')
