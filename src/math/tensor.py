# input x is generally [secid , time , feature] , the default dim is the most suitable for panel data
# dim means to eliminate the dimension , or along the dimension if the result will keep the dimension

import torch
import torch.nn.functional as F
from torch import Tensor , nan
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Literal

from .basic import alert_message , DIV_TOL , same , allna
    
def process_factor(value : Tensor | None , * , stream = 'inf_winsor_norm' , dim = 0 , trim_ratio = 7. , **kwargs):
    '''
    ------------------------ process factor value ------------------------
    处理因子值 , 'inf_trim_winsor_norm_neutral_nan'
    input:
        value:         factor value to be processed
        process_key:   can be any of 'inf_trim/winsor_norm_neutral_nan'
        dim:           default to 0 , to process cross-sectional data
        trim_ratio:    what extend can be identified as outlier? range is determined as med ± trim_ratio * brandwidth
        norm_tol:      if norm required, the tolerance to eliminate factor if standard deviation is too trivial
    output:
        value:         processed factor value
    '''
    if value is None or allna(value , inf_as_na = True): 
        return None
    
    # assert 'inf' in stream or 'trim' in stream or 'winsor' in stream , stream
    if 'trim' in stream or 'winsor' in stream:
        med       = value.nanquantile(0.5, dim=dim,keepdim = True , interpolation='lower')
        bandwidth = (value.nanquantile(0.75 , dim , keepdim=True) - value.nanquantile(0.25 , dim , keepdim=True)) / 2
        lbound , ubound = med - trim_ratio * bandwidth , med + trim_ratio * bandwidth
    for _str in stream.split('_'):
        if _str == 'mean':
            value -= torch.nanmean(value , dim, keepdim=True)
        elif _str == 'inf':
            value.nan_to_num_(torch.nan,torch.nan,torch.nan)
        elif _str == 'trim':
            value[(value > ubound) + (value < lbound)] = torch.nan
        elif _str == 'winsor':
            value = torch.where(value > ubound , ubound , value)
            value = torch.where(value < lbound , lbound , value)
        elif _str == 'norm': 
            value -= torch.nanmean(value , dim, keepdim=True)
            value /= value.square().nansum(dim , keepdim = True).sqrt() + 1e-6 
        elif _str == 'nan': 
            value = value.nan_to_num_()
    return value

def kthvalue_by_topk(x: Tensor, k: int, * , dim=-1, keepdim=True , largest=False):
    """
    Get the k-th smallest value by topk
    """
    # Get the k smallest elements
    vals, _ = torch.topk(x, k, dim=dim, largest=largest, sorted=True)
    # The k-th smallest is the last one in this sorted list
    res = vals.select(dim, -1)
    return res.unsqueeze(dim) if keepdim else res

class TsRoller:
    @staticmethod
    def unfold(x : Tensor , d : int , * , dim :int | Literal[1] = 1, nan = nan , pinf = torch.inf , ninf = -torch.inf, **kwargs):
        return x.nan_to_num(nan,pinf,ninf).unfold(dim,d,1)
    
    @staticmethod
    def fold(z : Tensor , d : int , * , dim : int = 1 , nan = nan , **kwargs):
        pad = tuple([0] * (z.ndim - dim - 1) * 2 + [d-1,0])
        return F.pad(z , pad , value = nan).nan_to_num(nan)

    @classmethod
    def decor(cls , n_arg : Literal[1,2] = 1, **decor_kwargs):
        if n_arg == 1:
            return cls.decorator_x(**decor_kwargs)
        elif n_arg == 2:
            return cls.decorator_xy(**decor_kwargs)
        else:
            raise ValueError(f'Invalid number of arguments: {n_arg}')

    @classmethod
    def decorator_x(cls , **decor_kwargs):
        def decorator(func):
            def wrapper(x : Tensor , d : int , *args , dim : int = 1 , **kwargs):
                x = cls.unfold(x , d , dim = dim , **decor_kwargs)
                z = func(x , 1 , *args , dim = -1 , **kwargs)
                z = cls.fold(z , d , dim = dim , **decor_kwargs)
                return z
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

    @classmethod
    def decorator_xy(cls , **decor_kwargs):
        def decorator(func):
            def wrapper(x : Tensor , y : Tensor , d : int , *args , dim : int = 1 , **kwargs):
                x = cls.unfold(x , d , dim = dim , **decor_kwargs)
                y = cls.unfold(y , d , dim = dim , **decor_kwargs)
                z = func(x , y , 1 , *args , dim = -1 , **kwargs)
                z = cls.fold(z , d , dim = dim , **decor_kwargs)
                return z
            wrapper.__name__ = func.__name__
            return wrapper
        return decorator

def nancount(x : torch.Tensor , * ,  dim=None, keepdim=False):  
    return x.isfinite().sum(dim = dim , keepdim = keepdim)

def nanmean(x : torch.Tensor , * , dim=None, keepdim=False):  
    try:
        return x.nanmean(dim = dim , keepdim = keepdim)
    except Exception:
        return x.nansum(dim = dim , keepdim = keepdim) / nancount(x , dim = dim , keepdim = keepdim)

def nanstd(x : torch.Tensor , * , dim=None, keepdim=False , correction=1):
    x = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = (torch.nansum(x ** 2 , dim , keepdim=keepdim) / (torch.sum(~x.isnan() ,dim = dim , keepdim=keepdim) - correction)).sqrt()
    return x_stddev

def nanmedian(x : torch.Tensor , dim=None, keepdim=False):
    if not x.is_floating_point():
        x = x.to(torch.float)
    return x.nanquantile(0.5, dim=dim,keepdim = keepdim , interpolation='lower')

def standardize(x : torch.Tensor, * , dim : int | None = 0):
    if x.isnan().all().item(): 
        return x
    x = (x - nanmean(x,dim=dim,keepdim=True)) / (nanstd(x,dim=dim,correction=0,keepdim=True) + DIV_TOL)
    return x

def rank(x : Tensor , * , dim : int | None = 0) -> Tensor:
    assert (len(x.shape) <= 3) , x.shape
    if dim is None:
        old_shape = x.shape
        x_rank = rank(x.flatten() , dim = 0).reshape(old_shape)
    else:
        x_rank = x.argsort(dim=dim).argsort(dim=dim).to(torch.float32) + 1 # .where(~x.isnan() , nan)
        x_rank[x.isnan()] = nan
    return x_rank

def rank_pct(x : Tensor , * , dim : int | None = 0) -> Tensor:
    assert (len(x.shape) <= 3) , x.shape
    if dim is None:
        old_shape = x.shape
        x_rank = rank_pct(x.flatten() , dim = 0).reshape(old_shape)
    else:
        x_rank = x.argsort(dim=dim).argsort(dim=dim).to(torch.float32) + 1 # .where(~x.isnan() , nan)
        x_rank[x.isnan()] = nan
        x_rank = x_rank / ((~x_rank.isnan()).sum(dim=dim, keepdim=True))
    return x_rank

def rankic_2d(x : Tensor , y : Tensor , * , dim : int | None = 0 , universe : Tensor | None = None , min_coverage = 0.5):
    valid = ~y.isnan()
    if universe is not None: 
        valid *= universe.nan_to_num(0).to(torch.bool)
    x = torch.where(valid , x , nan)

    coverage = (~x.isnan()).sum(dim=dim)
    x = rank_pct(x , dim = dim)
    y = rank_pct(y , dim = dim)
    ic = corrwith(x , y , dim=dim)
    return ic if ic is None else torch.where(coverage < min_coverage * valid.sum(dim=dim) , nan , ic)

def dummy(x : Tensor , * , ex_last = True):
    # will take a huge amount of memory, the suggestion is to use torch.cuda.empty_cache() after neutralization
    if not isinstance(x , Tensor): 
        x = torch.FloatTensor(x)
    xmax = x.nan_to_num().max().int().item() 
    dummy = x.nan_to_num(xmax + 1).to(torch.int64)
    # dummy = torch.where(x.isnan() , m + 1 , x).to(torch.int64)
    dummy = F.one_hot(dummy).to(torch.float)[...,:xmax+1-ex_last] # slightly faster but will take a huge amount of memory
    dummy = dummy[...,dummy.sum(dim = tuple(range(x.dim()))) != 0]
    return dummy

def concat_factors(*factors : Tensor , dim : int = -1 , device = None) -> Tensor:
    facs = [f for f in factors if f is not None]
    ts = torch.concat(facs , dim = dim)
    if device is not None:
        ts = ts.to(device = device)
    return ts

def concat_factors_2d(*factors : Tensor , dim : int = -1 , device = None) -> Tensor:
    facs = [f.unsqueeze(dim) if f.ndim == 2 else f for f in factors if f is not None]
    ts = torch.concat(facs , dim = dim)
    if device is not None:
        ts = ts.to(device = device)
    return ts

def neutralize_xdata_2d(x : Tensor | None , groups : None | list | tuple | Tensor = None):
    # not the most time efficient way, since its is very memory consuming

    if groups is None:
        groups = []
    elif not isinstance(groups , (list , tuple)): 
        groups  = [groups]
    if not groups and x is None:
        return None
    n_sample = x.shape[0] if x is not None else groups[0].shape[0]
    if x is None:
        x = torch.tensor([]).reshape(n_sample,0,0).to(groups[0].device)
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    for g in groups: 
        x = concat_factors(x , dummy(g , ex_last=True))
    x.nan_to_num_(nan , nan , nan)
    x = F.pad(x , (1,0) , value = 1.)
    return x

def betas_torch(x : Tensor , y : Tensor , * , method = ['lstsq', 'inv']) -> Tensor:
    x , y = x.float() , y.float()
    assert x.shape[0] == y.shape[0] , (x.shape , y.shape)
    b = None
    for m in method:
        try:
            if m == 'lstsq':
                b = torch.linalg.lstsq(x , y , rcond=None)[0]
            elif m == 'inv':
                M = x.T.mm(x)
                norm = M.norm()
                M_ = M / norm + torch.diag(torch.ones(len(M)) * 2e-5).to(x)
                b = (torch.linalg.inv(M_) / norm).mm(x).mm(y.T)
            else:
                raise ValueError(f'Invalid method: {m}')
            return b
        except Exception as e: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
            alert_message(f'neutralization error in betas_torch.{m}: {e}')

    return torch.zeros(x.shape[-1],1).to(x)

def betas_np(x : np.ndarray , y : np.ndarray , * , method = ['sk', 'lstsq', 'inv']) -> np.ndarray:
    assert x.shape[0] == y.shape[0] , (x.shape , y.shape)
    for m in method:
        try:
            if m == 'sk':
                b = LinearRegression(fit_intercept=False).fit(x, y).coef_.T
            elif m == 'lstsq':
                b = np.linalg.lstsq(x , y , rcond=None)[0]
            elif m == 'inv':
                M = x.T.dot(x)
                norm = np.linalg.norm(M)
                M_ = M / norm + np.diag(np.ones(len(M)) * 2e-5)
                b = (torch.linalg.inv(M_) / norm).mm(x).mm(y.T)
            else:
                raise ValueError(f'Invalid method: {m}')
            return b
        except Exception as e: # 20240215: numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares
            alert_message(f'neutralization error in betas_np.{m}: {e}')
    return np.zeros((x.shape[-1],1))

def beta_calculator(method = 'np'):
    assert method in ['torch' , 'np'] , method
    def wrapper(x : Tensor | np.ndarray , y : Tensor | np.ndarray , **kwargs):
        if method == 'np':
            assert isinstance(x , np.ndarray) and isinstance(y , np.ndarray) , (type(x) , type(y))
            return betas_np(x , y)
        elif method == 'torch':
            assert isinstance(x , Tensor) and isinstance(y , Tensor) , (type(x) , type(y))
            return betas_torch(x , y)
    return wrapper

def neutralize_2d(y : Tensor | None , x : Tensor | None , * ,
                  dim : int = 0 , method = 'np' , zscore = True , device = None , inplace = False , 
                  min_coverage = 3):  # [tensor (TS*C), tensor (TS*C)]
    if x is None or y is None or x.numel() == 0: 
        return y
    assert method in ['np' , 'torch'] , method
    assert dim in [0,1] , dim
    assert y.ndim == 2 , y.dim()
    assert x.ndim in [2,3] , x.dim()
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    old_device = y.device
    valid_date_secid  = x[...,0].isfinite() # [date , secid , factor]
    for k in range(1, x.shape[-1]): 
        valid_date_secid += x[...,k].isfinite()
    valid_date_secid *= y.isfinite() 
    x = x.nan_to_num(0,0,0)
    if inplace:
        y.nan_to_num_(nan , nan , nan).unsqueeze_(-1)
    else:
        y = y.nan_to_num(nan , nan , nan).unsqueeze(-1)
    
    if dim == 0: 
        # put date dimension to the first dimension
        x , y , valid_date_secid = x.permute(1,0,2) , y.permute(1,0,2) , valid_date_secid.permute(1,0)
    valid_feature = ~(x == 0).all(0) # [secid , factor]
    if device is not None:  
        x , y = x.to(device) , y.to(device)
    res = None
    if False and method == 'torch' and valid_date_secid.all() and valid_feature.all():
        # fastest, but cannot deal nan's which is always the case, so will not enter here
        try:
            model = torch.linalg.lstsq(x , y , rcond=None)
            res = (y - x @ model[0])
        except Exception:
            res = None
    if res is not None:
        y = res
    else: 
        betas_func = beta_calculator(method)
        if method == 'np': 
            xs , ys , dss , fs = x.cpu().numpy() , y.cpu().numpy() , valid_date_secid.cpu().numpy()  , valid_feature.cpu().numpy() 
        else:
            xs , ys , dss , fs = x , y , valid_date_secid , valid_feature
        for i , (y_ , x_ , s_ , k_) in enumerate(zip(ys , xs , dss , fs)):
            if s_.sum() < min_coverage or k_.sum() == 0: 
                continue
            betas = betas_func(x_[s_][:,k_] , y_[s_])
            ys[i,s_] -= (x_[s_][:,k_] @ betas) # type: ignore
        y = torch.FloatTensor(ys)
    if dim == 0: 
        y = y.permute(1,0,2)
    y.squeeze_(-1).to(old_device)
    if zscore:
        y = zscore_inplace(y , dim = dim)
    return y

def neutralize_1d(y : Tensor | None , x : Tensor | None , insample : Tensor | None , * , 
                  method = 'torch' , zscore = True , device = None , inplace = False , 
                  min_coverage = 3):  # [tensor (TS*C), tensor (TS*C)]
    if x is None or y is None or x.numel() == 0: 
        return y

    if insample is None:
        insample = torch.ones_like(y).to(torch.bool)

    assert method in ['np' , 'torch'] , method
    assert y.shape == insample.shape , (y.shape , insample.shape)
    assert y.dim() == 1 , y.dim()
    
    if x.dim() == 1: 
        x = x.reshape(-1,1)
    old_device = y.device
    valid_x  = x[:,0].isfinite()
    for k in range(1, x.shape[-1]): 
        valid_x += x[:,k].isfinite()
    valid_x *= y.isfinite() 
    if valid_x.sum() < min_coverage: 
        return y

    x = x.nan_to_num(0,0,0)
    if inplace:
        y.nan_to_num_(nan , nan , nan).unsqueeze_(-1)
    else:
        y = y.nan_to_num(nan , nan , nan).unsqueeze(-1)
    
    if device is not None:  
        x , y = x.to(device) , y.to(device)
    betas_func = beta_calculator(method)
    if method == 'np': 
        xs , ys , ins = x.cpu().numpy() , y.cpu().numpy() , (valid_x * insample).cpu().numpy()
    else:
        xs , ys , ins = x , y , valid_x * insample

    x_ , y_ , = xs[ins] , ys[ins]
    k_ = ~(x_ == 0).all(0)
    betas = betas_func(x_[:,k_] , y_)
    ys[ins] = ys[ins] - (x_[:,k_] @ betas) # type: ignore
    ys = torch.FloatTensor(ys).squeeze_(-1).to(old_device)
    if zscore:
        ys = zscore_inplace(ys , dim = 0)
    return ys

def corrwith(x : Tensor , y : Tensor , * , dim : int | None = 1):
    if same(x , y): 
        return torch.where(x.nansum(dim) > 2 , 1 , nan)
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  
    y_ymean = y - torch.nanmean(y, dim, keepdim=True) 
    cov  = torch.nansum(x_xmean * y_ymean, dim) 
    ssd  = x_xmean.square().nansum(dim).sqrt() * y_ymean.square().nansum(dim).sqrt()
    ssd[ssd == 0] = 1e-4
    corr = cov / ssd
    return corr

def covariance(x : Tensor , y : Tensor , * , dim : int | None = 1):
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  # [TS, C]
    y_ymean = y - torch.nanmean(y, dim, keepdim=True)  # [TS, C]
    cov = torch.nansum(x_xmean * y_ymean, dim)  # [TS, 1]
    return cov

def beta(x : Tensor , y : Tensor , * , dim : int | None = 1):
    if same(x , y): 
        return torch.where(torch.nansum(x , dim=dim) > 2 , 1 , nan)
    return covariance(x,y,dim=dim) / covariance(x,x,dim=dim)

def beta_pos(x : Tensor , y : Tensor , * , dim : int | None = 1):
    if same(x , y): 
        return torch.where(x.nansum(dim) > 2 , 1 , nan)
    y = torch.where(x < 0 , nan , y)
    x = torch.where(x < 0 , nan , x)
    return covariance(x,y,dim=dim) / covariance(x,x,dim=dim)

def beta_neg(x : Tensor , y : Tensor , * , dim : int | None = 1):
    if same(x , y): 
        return torch.where(x.nansum(dim) > 2 , 1 , nan)
    y = torch.where(x > 0 , nan , y)
    x = torch.where(x > 0 , nan , x)
    return covariance(x,y,dim=dim) / covariance(x,x,dim=dim)

def stddev(x : Tensor , * , dim : int | None = 1):
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)
    return torch.nansum(x_xmean ** 2, dim).sqrt()

def zscore(x : Tensor , * , dim : int | None = 0 , index : int | None = None):
    x_xmean  = x - torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = torch.nansum(x_xmean ** 2, dim , keepdim=index is None).sqrt()
    if index is not None: 
        x_xmean = x_xmean.select(dim,index)
    z = x_xmean / (x_stddev + 1e-4 * x_stddev.nanmean())
    return z

def abs(x):
    return torch.abs(x)

def zscore_inplace(x : Tensor , * , dim : int | None = 0):
    x -= torch.nanmean(x , dim, keepdim=True)  # [TS, C]
    x_stddev = nanstd(x , dim = dim , keepdim = True)
    x /= (x_stddev + 1e-4 * x_stddev.abs().nanmean())
    return x

def add(x : Tensor , y : Tensor):
    """x+y"""
    return x + y

def sub(x : Tensor , y : Tensor):
    """x-y"""
    return x - y

def mul(x : Tensor , y : Tensor):
    """x*y"""
    return x * y

def div(x : Tensor , y : Tensor):
    """x/y"""
    return x / y

def add_int(x : Tensor , d : int):
    """x+d"""
    return x + d

def sub_int1(x : Tensor , d : int):
    """x-d"""
    return x - d

def sub_int2(x : Tensor , d : int):
    """x-d"""
    return x - d

def mul_int(x : Tensor , d : int):
    """x*d"""
    return x * d

def div_int1(x : Tensor , d : int):
    """x/d"""
    return x / d

def div_int2(x : Tensor , d : int):
    """x/d"""
    return x / d

def neg(x : Tensor):
    """-x"""
    return -x

def neg_int(x : int):
    """-x"""
    return -x

def sigmoid(x : Tensor):
    """sigmoid(x)"""
    return 1 / (1 + torch.exp(-x))

def rank_sub(x : Tensor , y : Tensor , * , dim : int | None = 0):
    """rank_pct(x,dim) - rank_pct(y,dim)"""
    return rank_pct(x,dim=dim) - rank_pct(y,dim=dim)

def rank_add(x : Tensor , y : Tensor , * , dim : int | None = 0):
    """rank_pct(x,dim) + rank_pct(y,dim)"""
    return rank_pct(x,dim=dim) + rank_pct(y,dim=dim)

def rank_div(x : Tensor , y : Tensor , * , dim : int | None = 0):
    """rank_pct(x,dim) / rank_pct(y,dim)"""
    return rank_pct(x,dim=dim) / rank_pct(y,dim=dim)

def rank_mul(x : Tensor , y : Tensor , * , dim : int | None = 0):
    """rank_pct(x,dim) * rank_pct(y,dim)"""
    return rank_pct(x,dim=dim) * rank_pct(y,dim=dim)

def log(x : Tensor):
    """log(x)"""
    return x.log()

def sqrt(x : Tensor):
    """sqrt(x)"""
    return x.sqrt()

def square(x : Tensor):
    """square(x)"""
    return x.square()

def lin_decay(x : Tensor , * , dim = 1):   #only for rolling
    """d日衰减加权平均,加权系数为 d, d-1,...,1"""
    if dim < 0:
        dim = len(x.shape) + dim
    raw_shape = x.shape
    weight_shape = [1 if i != dim else raw_shape[dim] for i in range(len(raw_shape))]
    weight = torch.arange(1, raw_shape[dim] + 1, 1).reshape(weight_shape).to(x)
    return (x * weight).nansum(dim=dim) / ((weight * (~x.isnan())).sum(dim=dim))

def sign(x : Tensor):
    """sign(x)"""
    return x.sign()

def ts_delay(x : Tensor , d : int , * , dim : Literal[1] = 1, no_alert = False):
    """delay x by d"""
    if d > x.shape[dim]: 
        return x * nan
    if d < 0 and not no_alert: 
        alert_message('Beware! future information used!' , color = 'lightred')
    z = x.roll(d, dims=dim)
    if d >= 0:
        z[:,:d] = nan
    else:
        z[:,d:] = nan
    return z

def ts_delta(x : Tensor , d : int , * , dim : Literal[1] = 1, no_alert = False):
    """delta of d days"""
    if d > x.shape[0]: 
        return x * nan
    if d < 0 and not no_alert: 
        alert_message('Beware! future information used!' , color = 'lightred')
    z = x - ts_delay(x, d, dim=dim)
    return z

def scale(x : Tensor , c = 1 , * , dim = 0):
    """scale x by c along dim"""
    return c * x / x.abs().nansum(dim=dim, keepdim=True)

def signedpower(x : Tensor , a : float):
    """signed power of x by a"""
    return x.sign() * x.abs().pow(a)

def pctchg(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """percentage change of d days"""
    return (x - ts_delay(x,d , dim = dim)) / abs(ts_delay(x,d , dim = dim))

@TsRoller.decor(1)
def ts_zscore(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling ending z-score of d days"""
    return zscore(x , dim = dim , index = -1)

@TsRoller.decor(1)
def ts_mean(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling ending mean of d days (Moving Average)"""
    return torch.nanmean(x , dim = dim)

@TsRoller.decor(1 , nan = np.inf)
def ts_min(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling ending min of d days"""
    return torch.min(x , dim=dim).values

@TsRoller.decor(1 , nan = -np.inf)
def ts_max(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling ending max of d days"""
    return torch.max(x , dim=dim).values

@TsRoller.decor(1 , nan = np.inf)
def ts_argmin(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling ending argmin of d days"""
    return torch.argmin(x , dim=dim).to(torch.float)

@TsRoller.decor(1 , nan = -np.inf)
def ts_argmax(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling ending argmax of d days"""
    return torch.argmax(x , dim=dim).to(torch.float)

@TsRoller.decor(1)
def ts_rank(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling ending rank of d days"""
    return rank_pct(x,dim=dim)[...,-1]

@TsRoller.decor(1 , nan = 0)
def ts_stddev(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling ending stddev of d days"""
    return torch.std(x,dim=dim)

@TsRoller.decor(1 , nan = 0)
def ts_sum(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling sum of d days"""
    return torch.sum(x,dim=dim)

@TsRoller.decor(1 , nan = 1)
def ts_product(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling product of d days"""
    return torch.prod(x,dim=dim)

@TsRoller.decor(1)
def ts_lin_decay(x : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling lin_decay of d days"""
    return lin_decay(x , dim=dim)

def ts_decay_pos_dif(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling decay of positive difference of x and y of d days"""
    value = x - y
    value[value < 0] = 0
    return ts_lin_decay(value, d , dim=dim)

@TsRoller.decor(2,nan=0)
def ts_corr(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling correlation of x and y of d days"""
    return corrwith(x , y , dim=dim)

@TsRoller.decor(2 , nan = 0)
def ts_beta(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling beta of x and y of d days"""
    return beta(x , y , dim=dim)

@TsRoller.decor(2 , nan = 0)
def ts_beta_pos(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling beta of positive x and y of d days"""
    return beta_pos(x , y , dim=dim)

@TsRoller.decor(2 , nan = 0)
def ts_beta_neg(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling beta of negative x and y of d days"""
    return beta_neg(x , y , dim=dim)

@TsRoller.decor(2 , nan = 0)
def ts_cov(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling covariance of x and y of d days"""
    return covariance(x , y , dim=dim)

@TsRoller.decor(2,nan=0)
def ts_rankcorr(x : Tensor , y : Tensor , d : int , * , dim : Literal[1] = 1):
    """rolling rank correlation of x and y of d days"""
    return corrwith(rank_pct(x,dim=dim) , rank_pct(y,dim=dim) , dim=dim)

def conditional_x(
    x : Tensor , d : int , n : int , * ,
    dim : Literal[1] = 1, method : Literal['btm' , 'top' , 'diff'] , use : Literal['mean' , 'thres'] = 'mean',
    force_directional_sign : bool = False
):
    """
    conditional x by method and use
    e.g. method = 'btm', use = 'mean' means the bottom n x values in rolling d days are used to calculate the mean of x
    """
    assert method in ['btm' , 'top' , 'diff'] , method
    n = min(d, n)
    x = TsRoller.unfold(x , d , dim = dim)
    groups : list[Tensor | None] = [None , None]
    if method in ['btm' , 'diff']:
        condition = (x <= kthvalue_by_topk(x, n, dim=-1, keepdim=True, largest=False))
        if force_directional_sign: 
            condition *= (x < 0)
        value = torch.where(condition , x , nan).nan_to_num_(nan , nan , nan)
        groups[0] = value.nanmean(dim=-1) if use == 'mean' else value.nan_to_num(-torch.inf).max(dim=-1).values
    if method in ['top' , 'diff']:
        condition = (x >= kthvalue_by_topk(x, n, dim=-1, keepdim=True, largest=True))
        if force_directional_sign: 
            condition *= (x > 0)
        value = torch.where(condition , x , nan).nan_to_num_(nan , nan , nan)
        groups[1] = value.nanmean(dim=-1) if use == 'mean' else value.nan_to_num(-torch.inf).max(dim=-1).values
    
    z = [grp for grp in groups if grp is not None]
    if len(z) == 1:
        z = z[0]
    elif len(z) == 2:
        z = z[1] - z[0]
    else:
        raise ValueError(f'Invalid number of groups: {len(z)}')
    z = TsRoller.fold(z , d , dim = dim , nan = nan)
    return z

def conditional_y_on_x(
    x : Tensor , y : Tensor , d : int , n : int , * ,
    dim : Literal[1] = 1, method : Literal['btm' , 'top' , 'diff'] , use : Literal['mean' , 'thres'] = 'mean',
    force_directional_sign : bool = False
):
    """
    conditional y on x by method and use
    e.g. method = 'btm', use = 'mean' means the average y value of the n bottom x values in rolling d days
    """
    assert method in ['btm' , 'top' , 'diff'] , method
    n = min(d, n)
    x = TsRoller.unfold(x , d , dim = dim)
    y = TsRoller.unfold(y , d , dim = dim)
    groups : list[Tensor | None] = [None , None]
    if method in ['btm' , 'diff']:
        condition = (x <= kthvalue_by_topk(x, n, dim=-1, keepdim=True, largest=False))
        if force_directional_sign: 
            condition *= (x < 0)
        value = torch.where(condition , y , nan).nan_to_num_(nan , nan , nan)
        groups[0] = value.nanmean(dim=-1) if use == 'mean' else value.nan_to_num(-torch.inf).max(dim=-1).values

    if method in ['top' , 'diff']:
        condition = (x >= kthvalue_by_topk(x, n, dim=-1, keepdim=True, largest=True))
        if force_directional_sign: 
            condition *= (x > 0)
        value = torch.where(condition , y , nan).nan_to_num_(nan , nan , nan)
        groups[1] = value.nanmean(dim=-1) if use == 'mean' else value.nan_to_num(-torch.inf).max(dim=-1).values
        
    z = [grp for grp in groups if grp is not None]
    if len(z) == 1:
        z = z[0]
    elif len(z) == 2:
        z = z[1] - z[0]
    else:
        raise ValueError(f'Invalid number of groups: {len(z)}')
    z = TsRoller.fold(z , d , dim = dim , nan = nan)
    return z

def ts_btm_y_on_x(x : Tensor , y : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    '在过去d日上,根据x进行排序,取最小n个x的y的平均值'
    return conditional_y_on_x(x, y, d, n, dim=dim, method='btm')

def ts_top_y_on_x(x : Tensor , y : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    '在过去d日上,根据x进行排序,取最大n个x的y的平均值'
    return conditional_y_on_x(x, y, d, n, dim=dim, method='top')

def ts_dif_y_on_x(x : Tensor , y : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    '在过去d日上,根据x进行排序,取最大n个x的y的平均值与最小n个x的y的平均值的差值'
    return conditional_y_on_x(x, y, d, n, dim=dim, method='diff')

def ts_btm_x(x : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """在过去d日上,根据x进行排序,取最小n个x的平均值"""
    return conditional_x(x, d, n, dim=dim, method='btm')

def ts_top_x(x : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """在过去d日上,根据x进行排序,取最大n个x的平均值"""
    return conditional_x(x, d, n, dim=dim, method='top')

def ts_dif_x(x : Tensor , d : int , n : int , * , dim : Literal[1] = 1):
    """在过去d日上,根据x进行排序,取最大n个x的平均值与最小n个x的平均值的差值"""
    return conditional_x(x, d, n, dim=dim, method='diff')

def transpose_qkv(X : torch.Tensor , num_heads : int):
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X : torch.Tensor , num_heads : int):
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)