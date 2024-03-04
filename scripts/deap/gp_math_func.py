import pandas as pd
import numpy as np
from numba import jit , cuda

import torch
from torch import nn
from sklearn.linear_model import LinearRegression

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#import os
#if os.name == 'posix': assert(device == torch.device('cuda'))
#print(__name__)

nan = torch.nan
invalid = torch.Tensor()

def is_invalid(x):
    return x is invalid or (isinstance(x , torch.Tensor) and x.numel() == 0)

def allna(x , inf_as_na = False):
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

def prima_legitimate(check_valid = 1 , check_exact = True , check_allna = False , check_same = False):
    def decorator(func):
        def wrapper(*args , **kwargs):
            if check_valid >= 1:
                if is_invalid(args[0]): return invalid
                if check_allna and allna(args[0]): return invalid
            if check_valid >= 2:
                if is_invalid(args[1]): return invalid
                if check_allna and allna(args[1]): return invalid
                if check_exact and exact(args[0],args[1]): return invalid
                if check_same  and same(args[0],args[1]): return invalid
            v = func(*args , **kwargs)
            return v
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

def ts_roller(nan = nan , pinf = torch.inf , ninf = -torch.inf):
    def decorator(func):
        def wrapper(x , d , *args, **kwargs):
            assert d <= len(x) , (d,x)
            x = x.nan_to_num(nan,pinf,ninf)
            x = func(x.unfold(0,d,1) , d , *args, **kwargs)
            x = nn.functional.pad(x , [0,0,d-1,0] , value = nan)
            return x
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

def ts_coroller(nan = nan , pinf = torch.inf , ninf = -torch.inf):
    def decorator(func):
        def wrapper(x , y , d , *args, **kwargs):
            assert d <= len(x) , (d,x)
            x = x.nan_to_num(nan,pinf,ninf)
            y = y.nan_to_num(nan,pinf,ninf)
            z = func(x.unfold(0,d,1) , y.unfold(0,d,1) , d , *args, **kwargs)
            z = nn.functional.pad(z , [0,0,d-1,0] , value = nan)
            return z
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator

@prima_legitimate(2,False)
def rankic_2d(x , y , dim = 1 , universe = None , min_coverage = 0.5):
    valid = ~y.isnan()
    if universe is not None: valid *= universe.nan_to_num(False)
    x = torch.where(valid , x , nan)

    coverage = (~x.isnan()).sum(dim=dim)
    ic = corrwith(rank_pct(x), rank_pct(y) , dim=dim)
    if is_invalid(ic):
        return ic
    else:
        return torch.where(coverage < min_coverage * valid.sum(dim=dim) , nan , ic)

#%% 中性化函数
def one_hot(x):
    # will take a huge amount of memory, the suggestion is to use torch.cuda.empty_cache() after neutralization
    if not isinstance(x , torch.Tensor): x = torch.Tensor(x)
    m = x.nan_to_num().max().int().item() # type:ignore
    dummy = x.nan_to_num(m + 1).to(torch.int64)
    # dummy = torch.where(x.isnan() , m + 1 , x).to(torch.int64)
    dummy = torch.nn.functional.one_hot(dummy).to(torch.float)[...,:m+1] # slightly faster but will take a huge amount of memory
    dummy = dummy[...,dummy.sum(dim = tuple(range(x.dim()))) != 0]
    return dummy

def concat_factors(*factors , n_dims = 2 , dim = -1):
    if len(factors) == 0: return invalid
    if isinstance(factors , tuple): factors = list(factors)
    
    devtypes = [1 if fac.device.type == 'cuda' else 0 for fac in factors]
    device = factors[devtypes.index(max(devtypes))].device

    for i in range(len(factors)):
        if factors[i] is None: factors[i] = invalid
        if factors[i].dim() == n_dims:  factors[i] = factors[i].unsqueeze(dim)
        factors[i] = factors[i].to(device)

    if len(factors) == 1:
        factors = factors[0]  
    else:
        factors = torch.cat(factors , dim = dim)
    return factors.nan_to_num_(nan , nan , nan)

def neutralize_data(y , factors = None , groups = None):
    # not the most time efficient way, since its is very memory consuming
    assert y.dim() <= 2

    x = concat_factors(factors , n_dims = y.dim())

    if groups is None: groups  = []
    if not isinstance(groups , (list , tuple)): groups  = [groups]
    for g in groups:
        x = concat_factors(x , one_hot(g)[...,:-1] , n_dims = y.dim())
    if is_invalid(x): return x , y
    
    y = y.clone().nan_to_num_(nan , nan , nan).unsqueeze_(-1)
    torch.cuda.empty_cache()
    return x , y

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

def neutralize_2d(y , factors = None , group = None, dim = 1 , method = 'torch' , auto_intercept = True , silent= True , device = None):  # [tensor (TS*C), tensor (TS*C)]
    assert method in ['sk' , 'np' , 'torch']
    assert dim in [-1,0,1] , dim
    original_device = y.device
    x , y = neutralize_data(y , factors , group)
    if is_invalid(x): return y.squeeze_(-1)
    finite_ij   = y.isfinite().any(-1) * x.isfinite().any(-1)
    x = x.nan_to_num_(0,0,0)
    x = torch.nn.functional.pad(x , (1,0) , value = 1.)
    nonzero_ik = x.count_nonzero(dim).bool()
    if device is not None:  x , y = x.to(device) , y.to(device)
    if dim == 0: x , y = x.permute(1,0,2) , y.permute(1,0,2)
    if False and method == 'torch' and finite_ij.all() and nonzero_ik.all():
        # fastest, but cannot deal nan's which is always the case, so will not enter here
        model = torch.linalg.lstsq(x , y , rcond=None)
        y = (y - x @ model[0])
    else:
        betas_func = beta_calculator(method)
        
        if method in ['sk' , 'np']:
            y = y.cpu().numpy()
            x = x.cpu().numpy()
        y = y.squeeze_(-1)
        for i , (y_ , x_ , j_ , k_) in enumerate(zip(y , x , finite_ij , nonzero_ik)):
            if not silent and i % 500 == 0: print('neutralize by tradedate',i)
            #_nnan  = ~(y_.isnan().any(dim=-1) + x_.isnan().any(dim=-1))
            #if _nnan.sum() < 10:  continue
            
            #x_.nan_to_num_(0,0,0)
            #x_ = x_[:,(x_ != 0).any(0)]
            #betas = betas_func(x_[_nnan] , y_[_nnan])
            if j_.sum() < 10: continue
            x_ = x_[:,k_]
            betas = betas_func(x_[j_] , y_[j_])
            y[i] = (y_ - x_ @ betas).flatten()
        if isinstance(y , np.ndarray): y = torch.Tensor(y)
    if dim == 0: y = y.permute(1,0)
    y = zscore(y , dim = dim)
    torch.cuda.empty_cache()
    y = y.to(original_device)
    return y

#%% 相关系数函数
# 将以上函数以矩阵形式改写
@prima_legitimate(2,False)
def corrwith(x,y,dim=None):
    if same(x , y): return torch.where(x.nansum(dim) > 2 , 1 , nan)
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  # [TS, C]
    y_ymean = y - torch.nanmean(y, dim, keepdim=True)  # [TS, C]
    cov  = torch.nansum(x_xmean * y_ymean, dim) 
    xsd  = x_xmean.square().nansum(dim).sqrt() 
    ysd  = y_ymean.square().nansum(dim).sqrt()
    tol  = cov.nanmean() * 1e-4
    corr = cov / (xsd + tol) / (ysd + tol)
    return corr

@prima_legitimate(2,False)
def covariance(x,y,dim=None):
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  # [TS, C]
    y_ymean = y - torch.nanmean(y, dim, keepdim=True)  # [TS, C]
    cov = torch.nansum(x_xmean * y_ymean, dim)  # [TS, 1]
    return cov

@prima_legitimate(2,False)
def beta(x,y,dim=None):
    if same(x , y): return torch.where(x.nansum(dim) > 2 , 1 , nan)
    return covariance(x,y,dim) / covariance(x,x,dim)

@prima_legitimate(2,False)
def beta_pos(x,y,dim=None):
    if same(x , y): return torch.where(x.nansum(dim) > 2 , 1 , nan)
    y = torch.where(x < 0 , nan , y)
    x = torch.where(x < 0 , nan , x)
    return covariance(x,y,dim) / covariance(x,x,dim)

@prima_legitimate(2,False)
def beta_neg(x,y,dim=None):
    if same(x , y): return torch.where(x.nansum(dim) > 2 , 1 , nan)
    y = torch.where(x > 0 , nan , y)
    x = torch.where(x > 0 , nan , x)
    return covariance(x,y,dim) / covariance(x,x,dim)

@prima_legitimate(1)
def stddev(x , dim = 0 , keepdim = True):
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    return torch.nansum(x_xmean ** 2, dim , keepdim = keepdim).sqrt()

@prima_legitimate(1)
def zscore(x , dim = 0 , index = None):
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = torch.nansum(x_xmean ** 2, dim , keepdim=index is None).sqrt()
    if index: x_xmean = x_xmean.select(dim,index)
    z = x_xmean / (x_stddev + 1e-4 * x_stddev.nanmean())
    return z

#%% 其他函数
@prima_legitimate(2)
def add(x, y):
    'x+y'
    return x + y

@prima_legitimate(2)
def sub(x, y):
    'x-y'
    return x - y

@prima_legitimate(2)
def mul(x, y):
    'x*y'
    return x * y

@prima_legitimate(2)
def div(x, y):
    'x/y'
    return x / y

@prima_legitimate(1)
def add_int(x, d):
    'x+d'
    return x + d

@prima_legitimate(1)
def sub_int1(x, d):
    'x-d'
    return x - d

@prima_legitimate(1)
def sub_int2(x, d):
    'x-d'
    return x - d

@prima_legitimate(1)
def mul_int(x, d):
    'x*d'
    return x * d

@prima_legitimate(1)
def div_int1(x, d):
    'x/d'
    return x / d

@prima_legitimate(1)
def div_int2(x, d):
    'x/d'
    return x / d

def neg(x):
    '-x'
    return invalid if is_invalid(x) else -x

def neg_int(x):
    '-x'
    return -x

@prima_legitimate(1)
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

@prima_legitimate(2)
def rank_sub(x,y):
    return rank_pct(x,1) - rank_pct(y,1)

@prima_legitimate(2)
def rank_add(x,y):
    return rank_pct(x,1) + rank_pct(y,1)

@prima_legitimate(2)
def rank_div(x,y):
    return rank_pct(x,1) / rank_pct(y,1)

@prima_legitimate(2)
def rank_mul(x,y):
    return rank_pct(x,1) * rank_pct(y,1)

@prima_legitimate(1)
def log(x):
    return x.log()

@prima_legitimate(1)
def sqrt(x):
    return x.sqrt()

@prima_legitimate(1)
def square(x):
    return x.square()

@prima_legitimate(1)
def rank_pct(x,dim=1):
    assert (len(x.shape) <= 3)
    x_rank = x.argsort(dim=dim).argsort(dim=dim).to(torch.float32) + 1 # .where(~x.isnan() , nan)
    x_rank[x.isnan()] = nan
    x_rank = x_rank / ((~x_rank.isnan()).sum(dim=dim, keepdim=True))
    return x_rank

@prima_legitimate(1)
def lin_decay(x , dim = 0):   #only for rolling
    'd日衰减加权平均，加权系数为 d, d-1,...,1'
    shape_ = [1] * len(x.shape)
    shape_[dim] = x.shape[dim]
    coef = torch.arange(1, x.shape[dim] + 1, 1).reshape(shape_).to(x)
    return (x * coef).nansum(dim=dim) / ((coef * (~x.isnan())).sum(dim=dim))

@prima_legitimate(2)
def ts_decay_pos_dif(x, y, d):
    value = x - y
    value[value < 0] = 0
    return ts_lin_decay(value, d)

@prima_legitimate(1)
def sign(x):
    return x.sign()

@prima_legitimate(1)
def ts_delay(x, d):
    if d > x.shape[0]: return invalid
    if d < 0: print('Beware! future information used!')
    z = x.roll(d, dims=0)
    if d >= 0:
        z[:d,:] = np.nan
    else:
        z[d:,:] = np.nan
    return z

@prima_legitimate(1)
def ts_delta(x, d):
    if d > x.shape[0]: return invalid
    if d < 0: print('Beware! future information used!')
    z = x - ts_delay(x, d)
    return z

@prima_legitimate(1)
def scale(x, c = 1 , dim = 1):
    return c * x / x.abs().nansum(axis=dim, keepdim=True)

@prima_legitimate(1)
def signedpower(x, a):
    return x.sign() * x.abs().pow(a)

@prima_legitimate(1)
@ts_roller()
def ts_zscore(x, d):
    return zscore(x , dim = -1 , index = -1)

@prima_legitimate(1)
@ts_roller()
def ma(x, d):
    return torch.nanmean(x , dim=-1)

@prima_legitimate(1)
def pctchg(x,d):
    return (x - ts_delay(x,d)) / ts_delay(x,d).abs()

@prima_legitimate(1)
@ts_roller(np.inf)
def ts_min(x, d):
    return torch.min(x , dim=-1)[0]

@prima_legitimate(1)
@ts_roller(-np.inf)
def ts_max(x, d):
    return torch.max(x , dim=-1)[0]

@prima_legitimate(1)
@ts_roller(np.inf)
def ts_argmin(x, d):
    return torch.argmin(x , dim=-1).to(torch.float)

@prima_legitimate(1)
@ts_roller(-np.inf)
def ts_argmax(x, d):
    return torch.argmax(x , dim=-1).to(torch.float)

@prima_legitimate(1)
@ts_roller()
def ts_rank(x, d):
    return rank_pct(x,dim=-1)[...,-1]

@prima_legitimate(1)
@ts_roller(0)
def ts_stddev(x, d):
    return torch.std(x,dim=-1)

@prima_legitimate(1)
@ts_roller(0)
def ts_sum(x, d):
    return torch.sum(x,dim=-1)

@prima_legitimate(1)
@ts_roller(1)
def ts_product(x, d):
    return torch.prod(x,dim=-1)

@prima_legitimate(1)
@ts_roller()
def ts_lin_decay(x, d):
    return lin_decay(x , dim=-1)

@prima_legitimate(2)
@ts_coroller(0)
def ts_corr(x , y , d):
    return corrwith(x , y , dim=-1)

@prima_legitimate(2)
@ts_coroller(0)
def ts_beta(x , y , d):
    return beta(x , y , dim=-1)

@prima_legitimate(2)
@ts_coroller(0)
def ts_beta_pos(x , y , d):
    return beta_pos(x , y , dim=-1)

@prima_legitimate(2)
@ts_coroller(0)
def ts_beta_neg(x , y , d):
    return beta_neg(x , y , dim=-1)

@prima_legitimate(2)
@ts_coroller(0)
def ts_cov(x , y , d):
    return covariance(x , y , dim=-1)

@prima_legitimate(2)
@ts_coroller(0)
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
        
    z = [torch.where(grp , y , nan).nanmean(dim=-1) for grp in groups if grp is not None]
    z = torch.nn.functional.pad(z[0] if len(z) == 1 else (z[1] - z[0]) , [0,0,d-1,0] , value = nan)
    return z

@prima_legitimate(2)
def ts_xbtm_yavg(x, y, d, n):
    '在过去d日上，根据x进行排序，取最小n个x的y的平均值'
    return rlbxy(x, y, d, n, btm='btm')

@prima_legitimate(2)
def ts_xtop_yavg(x, y, d, n):
    '在过去d日上，根据x进行排序，取最大n个x的y的平均值'
    return rlbxy(x, y, d, n, btm='top')

@prima_legitimate(2)
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
        
    z = [torch.where(grp , x , nan).nanmean(dim=-1) for grp in groups if grp is not None]
    z = torch.nn.functional.pad(z[0] if len(z) == 1 else (z[1] - z[0]) , [0,0,d-1,0] , value = nan)
    return z

@prima_legitimate(1)
def ts_btm_avg(x, d, n):
    '在过去d日上，根据x进行排序，取最小n个x的y的平均值'
    return rlbx(x, d, n, btm='btm')

@prima_legitimate(1)
def ts_top_avg(x, d, n):
    '在过去d日上，根据x进行排序，取最大n个x的y的平均值'
    return rlbx(x, d, n, btm='top')

@prima_legitimate(1)
def ts_rng_dif(x, d, n):
    '在过去d日上，根据x进行排序，取最大n个x的y的平均值与最小n个x的y的平均值的差值'
    return rlbx(x, d, n, btm='diff')
