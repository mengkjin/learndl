import torch
import numpy as np

from scipy import stats

from .basic import DIV_TOL
from .tensor import rank_pct , rank

def rank_weight(x : torch.Tensor , dim = None):    
    r = rank(x , dim = dim)
    n = (~r.isnan()).sum(dim=dim,keepdim=True)
    p = (n - 1 - r) * 2 / (n - 1)
    w = torch.pow(0.5,p)
    return w / w.sum()

def demean(x : torch.Tensor, w : torch.Tensor | float, dim = None):
    return x - (w * x).mean(dim=dim,keepdim=True)

def pearson(x : torch.Tensor, y : torch.Tensor , w = None, dim = None , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    x1 , y1 = demean(x , w , dim) , demean(y , w , dim)
    return (w * x1 * y1).mean(dim = dim) / ((w * x1.square()).mean(dim=dim).sqrt() + DIV_TOL) / ((w * y1.square()).mean(dim=dim).sqrt() + DIV_TOL)

def ccc(x : torch.Tensor , y : torch.Tensor , w = None, dim = None , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    x1 , y1 = demean(x , w , dim) , demean(y , w , dim)
    cov_xy = (w * x1 * y1).mean(dim=dim)
    mse_xy = (w * (x1 - y1).square()).mean(dim=dim)
    return (2 * cov_xy) / (mse_xy + 2 * cov_xy + DIV_TOL)

def mse(x : torch.Tensor , y : torch.Tensor , w = None, dim = None , reduction='mean' , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    f = torch.mean if reduction == 'mean' else torch.sum
    return f(w * (x - y).square() , dim=dim)

def spearman(x : torch.Tensor , y : torch.Tensor , w = None , dim = None , **kwargs):
    x , y = rank(x , dim = dim) , rank(y , dim = dim)
    return pearson(x , y , w , dim , **kwargs)

def wpearson(x : torch.Tensor , y : torch.Tensor , dim = None , **kwargs):
    w = rank_weight(y , dim = dim)
    return pearson(x,y,w,dim)

def wccc(x : torch.Tensor , y : torch.Tensor , dim = None , **kwargs):
    w = rank_weight(y , dim = dim)
    return ccc(x,y,w,dim)

def wmse(x : torch.Tensor , y : torch.Tensor , dim = None , reduction='mean' , **kwargs):
    w = rank_weight(y , dim = dim)
    return mse(x,y,w,dim,reduction)

def wspearman(x : torch.Tensor , y : torch.Tensor , dim = None , **kwargs):
    w = rank_weight(y , dim = dim)
    return spearman(x,y,w,dim)

def np_drop_na(x : np.ndarray , y : np.ndarray):
    pairwise_nan = np.isnan(x) + np.isnan(y)
    x , y = x[~pairwise_nan] , y[~pairwise_nan]
    return x , y

def np_ic(x : np.ndarray , y : np.ndarray):
    x , y = np_drop_na(x,y)
    try:
        return stats.pearsonr(x,y)[0]
    except Exception:
        return np.nan
    
def np_rankic(x : np.ndarray , y : np.ndarray):
    x , y = np_drop_na(x,y)
    try:
        return stats.spearmanr(x,y)[0]
    except Exception:
        return np.nan
    
def np_ic_2d(x : np.ndarray , y : np.ndarray , dim=0):
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - np.nanmean(x, dim, keepdims=True)  
    y_ymean = y - np.nanmean(y, dim, keepdims=True) 
    cov  = np.nansum(x_xmean * y_ymean, dim) 
    ssd  = (np.nansum(np.square(x_xmean), dim) ** 0.5) * (np.nansum(np.square(y_ymean), dim) ** 0.5 )
    ssd[ssd == 0] = 1e-4
    corr = cov / ssd
    return corr

def np_rankic_2d(x : np.ndarray , y : np.ndarray , dim = 0):
    if dim == 0:
        return np.array([np_rankic(x[:,i],y[:,i]) for i in range(x.shape[1])])
    else:
        return np.array([np_rankic(x[i,:],y[i,:]) for i in range(x.shape[0])])

def ic(x : torch.Tensor , y : torch.Tensor):
    return ic_2d(x.flatten() , y.flatten() , 0)

def rankic(x : torch.Tensor , y : torch.Tensor):
    return rankic_2d(x.flatten() , y.flatten() , 0)

def ic_2d(x : torch.Tensor , y : torch.Tensor , dim=0):
    x = x + y * 0
    y = y + x * 0
    x_xmean = x - torch.nanmean(x, dim, keepdim=True)  
    y_ymean = y - torch.nanmean(y, dim, keepdim=True) 
    cov  = torch.nansum(x_xmean * y_ymean, dim) 
    ssd  = x_xmean.square().nansum(dim).sqrt() * y_ymean.square().nansum(dim).sqrt()
    ssd[ssd == 0] = 1e-4
    corr = cov / ssd
    return corr

def rankic_2d(x : torch.Tensor, y : torch.Tensor , dim = 1 , universe = None , min_coverage = 0.5):
    valid = ~y.isnan()
    if universe is not None: 
        valid *= universe.nan_to_num(False)
    x = torch.where(valid , x , torch.nan)

    coverage = (~x.isnan()).sum(dim=dim)
    x = rank_pct(x , dim = dim)
    y = rank_pct(y , dim = dim)
    ic = ic_2d(x , y , dim=dim)
    return ic if ic is None else torch.where(coverage < min_coverage * valid.sum(dim=dim) , torch.nan , ic)