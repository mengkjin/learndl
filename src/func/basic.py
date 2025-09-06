import os , pprint , psutil , torch
import numpy as np

from pytimedinput import timedInput
from scipy import stats
from torch import Tensor
from typing import Optional

DIV_TOL = 1e-4

def average_params(params_list : tuple[dict] | list[dict]):
    n = len(params_list)
    if n == 1: return params_list[0]
    new_params = {k:v / n for k,v in params_list[0].items()}
    for i, params in enumerate(params_list[1:]):
        for k, v in params.items():
            if k not in new_params.keys(): 
                raise KeyError(f'the {i}-th model has different params named {k}')
            else:
                new_params[k] += v / n
    return new_params

def emphasize_header(header=''):
    print('{: ^100}'.format(''))
    print('{:*^100}'.format(''))
    print('{:*^100}'.format('    '+header+'    '))
    print('{:*^100}'.format(''))
    print('{: ^100}'.format(''))
        
def tensor_nancount(x : Tensor , dim=None, keepdim=False):  
    return x.isfinite().sum(dim = dim , keepdim = keepdim)

def tensor_nanmean(x : Tensor , dim=None, keepdim=False):  
    try:
        return x.nanmean(dim = dim , keepdim = keepdim)
    except:
        return x.nansum(dim = dim , keepdim = keepdim) / tensor_nancount(x , dim = dim , keepdim = keepdim)

def tensor_nanstd(x : Tensor , dim=None, keepdim=False , correction=1):
    if dim is None: return torch.tensor(np.nanstd(x.flatten()))
    nancount = tensor_nancount(x , dim = dim , keepdim = True) - correction
    return ((x - tensor_nanmean(x , dim = dim , keepdim = True)).square() / nancount).nansum(dim = dim , keepdim = keepdim).sqrt()

def tensor_nanmedian(x : Tensor , dim=None, keepdim=False):
    if x.is_floating_point():
        return x.nanmedian() if dim is None else x.nanmedian(dim = dim , keepdim = keepdim).values
    else:
        return tensor_nanmedian(x.to(torch.float) , dim , keepdim).to(x.dtype)
        
def tensor_standardize_and_weight(x : Tensor, dim = None , weight_scheme = None):
    if x.isnan().all().item(): return (x , None) 
    x = (x - tensor_nanmean(x,dim=dim,keepdim=True)) / (tensor_nanstd(x,dim=dim,correction=0,keepdim=True) + DIV_TOL)
    if weight_scheme is None or weight_scheme == 'equal':
        w = None
    elif weight_scheme == 'top':
        w = torch.ones_like(x)
        try: 
            w[x > tensor_nanmedian(x , dim , True)] = 2
        except:    
            w[x > tensor_nanmedian(x)] = 2
    else:
        raise KeyError(weight_scheme)
    return (x, w)

def standardize_x(x : np.ndarray , dim=None):
    if np.isnan(x).all():
        ...
    elif dim is None or len(x.shape) == 1:
        x = (x - np.nanmean(x)) / (np.nanstd(x) + DIV_TOL)
    else:
        tran_dim = np.arange(len(x.shape))
        tran_dim[0],tran_dim[dim] = dim,0
        y = x.transpose(*tran_dim).reshape(x.shape[dim],-1) * 1.
        for i in range(y.shape[-1]):
            y[:,i] = standardize_x(y[:,i])
        x = y.reshape(*[x.shape[j] for j in tran_dim]).transpose(*tran_dim)
    return x

def standardize_and_weight(x : np.ndarray , dim=None):
    if np.isnan(x).all():
        ...
    elif dim is None or len(x.shape) == 1:
        x = (x - np.nanmean(x)) / (np.nanstd(x) + DIV_TOL)
        w = np.ones_like(x)
        w[x >= np.nanmedian(x)] = 2.
    else:
        tran_dim = np.arange(len(x.shape))
        tran_dim[0],tran_dim[dim] = dim,0
        y = x.transpose(*tran_dim).reshape(x.shape[dim],-1) * 1.
        w = np.ones_like(y)
        for i in range(y.shape[-1]):
            _x , _w = standardize_and_weight(y[:,i])
            y[:,i] , w[:,i] = _x , _w
        x = y.reshape(*[x.shape[j] for j in tran_dim]).transpose(*tran_dim)
        w = w.reshape(*[x.shape[j] for j in tran_dim]).transpose(*tran_dim)
    return x , w

def multi_bin_label(x : np.ndarray | Tensor , n = 10):
    y , w = np.zeros_like(x) , np.zeros_like(x)
    for i in range(n):
        low , high = np.quantile(x, i/n) , np.quantile(x, (i+1)/n)
        if i == n-1:
            y[(x >= low)] = 2 * i - n + 1
        elif i == 0:
            y[(x < high)] = 2 * i - n + 1
        else:
            y[(x >= low) & (x < high)] = 2 * i - n + 1
    w[:] = np.abs(y)
    return y, w


def bin_label(x : np.ndarray | Tensor):
    y , w = np.zeros_like(x) , np.zeros_like(x)
    y[x >= np.nanmedian(x)] = 1
    w[:] = y + 1
    return y, w

def tensor_rank(x : Tensor):    
    assert x.dim() == 1 , x.dim()
    # faster than x.argsort().argsort().to(x) for longer x
    # return torch.zeros_like(x).index_copy_(0,x.argsort(),torch.arange(len(x)).to(x))
    y = torch.zeros_like(x)
    y[x.argsort()] = torch.arange(len(x)).to(x)
    return y

def rank_weight(x : Tensor):    
    r = tensor_rank(x)
    w = torch.pow(0.5,((r.numel() - 1 - r) * 2 / (r.numel() - 1)))
    return w / w.sum()

def nd_rank(x : Tensor , dim = None):
    if dim is None:
        w = tensor_rank(x.flatten()).reshape(x.shape)
    else:
        if x.numel() == 0: return x
        w = x.clone().transpose(-1 , dim)
        new_shape = w.shape
        w = w.reshape(-1 , new_shape[-1])
        for i in range(len(w)): w[i] = tensor_rank(w[i])
        w = w.reshape(*new_shape).transpose(-1 , dim)   
    return w
def nd_rank_weight(x : Tensor , dim = None):
    if dim is None:
        w = rank_weight(x.flatten()).reshape(x.shape)
    else:
        if x.numel() == 0: return x
        w = x.clone().transpose(-1 , dim)
        new_shape = w.shape
        w = w.reshape(-1 , new_shape[-1])
        for i in range(len(w)): w[i] = rank_weight(w[i])
        w = w.reshape(*new_shape).transpose(-1 , dim)   
    return w 
def nd_minus_mean(x : Tensor, w : Tensor | float, dim = None):
    return x - (w * x).mean(dim=dim,keepdim=True)

def pearson(x : Tensor, y : Tensor , w = None, dim = None , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    x1 , y1 = nd_minus_mean(x , w , dim) , nd_minus_mean(y , w , dim)
    return (w * x1 * y1).mean(dim = dim) / ((w * x1.square()).mean(dim=dim).sqrt() + DIV_TOL) / ((w * y1.square()).mean(dim=dim).sqrt() + DIV_TOL)

def ccc(x : Tensor , y : Tensor , w = None, dim = None , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    x1 , y1 = nd_minus_mean(x , w , dim) , nd_minus_mean(y , w , dim)
    cov_xy = (w * x1 * y1).mean(dim=dim)
    mse_xy = (w * (x1 - y1).square()).mean(dim=dim)
    return (2 * cov_xy) / (mse_xy + 2 * cov_xy + DIV_TOL)

def mse(x : Tensor , y : Tensor , w = None, dim = None , reduction='mean' , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    f = torch.mean if reduction == 'mean' else torch.sum
    return f(w * (x - y).square() , dim=dim)

def spearman(x : Tensor , y : Tensor , w = None , dim = None , **kwargs):
    x , y = nd_rank(x , dim = dim) , nd_rank(y , dim = dim)
    return pearson(x , y , w , dim , **kwargs)

def wpearson(x : Tensor , y : Tensor , dim = None , **kwargs):
    w = nd_rank_weight(y , dim = dim)
    return pearson(x,y,w,dim)

def wccc(x : Tensor , y : Tensor , dim = None , **kwargs):
    w = nd_rank_weight(y , dim = dim)
    return ccc(x,y,w,dim)

def wmse(x : Tensor , y : Tensor , dim = None , reduction='mean' , **kwargs):
    w = nd_rank_weight(y , dim = dim)
    return mse(x,y,w,dim,reduction)

def wspearman(x : Tensor , y : Tensor , dim = None , **kwargs):
    w = nd_rank_weight(y , dim = dim)
    return spearman(x,y,w,dim)

def transpose_qkv(X : Tensor , num_heads : int):
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X : Tensor , num_heads : int):
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)

def np_drop_na(x : np.ndarray , y : np.ndarray):
    pairwise_nan = np.isnan(x) + np.isnan(y)
    x , y = x[~pairwise_nan] , y[~pairwise_nan]
    return x , y

def np_ic(x : np.ndarray , y : np.ndarray):
    x , y = np_drop_na(x,y)
    try:
        return stats.pearsonr(x,y)[0]
    except:
        return np.nan
    
def np_rankic(x : np.ndarray , y : np.ndarray):
    x , y = np_drop_na(x,y)
    try:
        return stats.spearmanr(x,y)[0]
    except:
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
    
def ts_rank_pct(x : torch.Tensor ,dim=-1):
    assert (len(x.shape) <= 3)
    x_rank = x.argsort(dim=dim).argsort(dim=dim).to(torch.float32) + 1
    x_rank[x.isnan()] = torch.nan
    x_rank = x_rank / ((~x_rank.isnan()).sum(dim=dim, keepdim=True))
    return x_rank

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
    if universe is not None: valid *= universe.nan_to_num(False)
    x = torch.where(valid , x , torch.nan)

    coverage = (~x.isnan()).sum(dim=dim)
    x = ts_rank_pct(x , dim = dim)
    y = ts_rank_pct(y , dim = dim)
    ic = ic_2d(x , y , dim=dim)
    return ic if ic is None else torch.where(coverage < min_coverage * valid.sum(dim=dim) , torch.nan , ic)

def total_memory(unit = 1e9):
    return psutil.Process(os.getpid()).memory_info().rss / unit

def match_values(values , src_arr , ambiguous = 0):
    if not isinstance(values , np.ndarray): 
        if isinstance(values , torch.Tensor): values = values.cpu().numpy()
        else: values = np.asarray(values)
    sorter = np.argsort(src_arr)
    index = np.tile(len(src_arr) , values.shape)
    if ambiguous == 0:
        index[np.isin(values , src_arr)] = sorter[np.searchsorted(src_arr, values[np.isin(values , src_arr)], sorter=sorter)]
    else:
        index[values <= max(src_arr)] = sorter[np.searchsorted(src_arr, values[values <= max(src_arr)], sorter=sorter)]
    return index

def merge_data_2d(data_tuple , row_tuple , col_tuple , row_all = None , col_all = None):
    if all([not isinstance(inp,tuple) for inp in (data_tuple , row_tuple , col_tuple)]):
        return data_tuple , row_tuple , col_tuple
    elif not all([isinstance(inp,tuple) for inp in (data_tuple , row_tuple , col_tuple)]):
        raise Exception(f'Not All of data_tuple , row_tuple , col_tuple are tuple instance!')
    
    assert len(data_tuple) == len(row_tuple) == len(col_tuple)
    for i in range(len(data_tuple)):
        #print(i , data_tuple[i].shape , (len(row_tuple[i]) , len(col_tuple[i])))
        assert data_tuple[i].shape == (len(row_tuple[i]) , len(col_tuple[i]))
    
    row_all = sorted(list(set().union(*row_tuple))) if row_all is None else row_all
    row_index = [[list(row_all).index(r) for r in row_i] for row_i in row_tuple]
    
    col_all = sorted(list(set().union(*col_tuple))) if col_all is None else col_all
    col_index = [[list(col_all).index(c) for c in col_i] for col_i in col_tuple]
    
    data_all = np.full((len(row_all) , len(col_all)) , np.nan)
    for i , data in enumerate(data_tuple):
        data_all[np.repeat(row_index[i],len(col_index[i])),np.tile(col_index[i],len(row_index[i]))] = data[:].flatten()
    return data_all , row_all , col_all
        
def list_converge(l , n = None , eps = 1e-6):
    '''Last n element of l has range smaller than eps'''
    if n is None: return max(l) - min(l) < eps   
    return len(l) >= n and (max(l[-n:]) - min(l[-n:]) < eps)

def pretty_print_dict(dictionary , width = 140 , sort_dicts = False):
    pprint.pprint(dictionary, indent = 1, width = width , sort_dicts = sort_dicts)

def subset(x , i):
    if isinstance(x , (list,tuple)):
        return type(x)([v[i] for v in x])
    elif isinstance(x , dict):
        return {k:v if v is None else v[i] for k,v in x.items()}
    else:
        return x[i]
    
def forward_fillna(arr , axis = 0):
    shape = arr.shape
    if axis < 0: axis = len(shape) + axis
    new_axes  = list(range(len(arr.shape)))
    new_axes[0] , new_axes[axis] = axis , 0
    
    arr = np.transpose(arr , new_axes)
    new_shape = arr.shape

    arr = arr.reshape(shape[axis],-1).transpose(1,0)
    idx = np.where(np.isnan(arr) == 0 , np.arange(arr.shape[1]), 0)
    idx = np.maximum.accumulate(idx, axis=1)
    out = arr[np.arange(idx.shape[0])[:,None], idx].transpose(1,0)
    out = np.transpose(out.reshape(new_shape) , new_axes)
    return out

def backward_fillna(arr, axis = 0):
    shape = arr.shape
    if axis < 0: axis = len(shape) + axis
    new_axes  = list(range(len(arr.shape)))
    new_axes[0] , new_axes[axis] = axis , 0

    arr = np.transpose(arr , new_axes)
    new_shape = arr.shape

    arr = arr.reshape(shape[axis],-1).transpose(1,0)
    idx = np.where(np.isnan(arr) == 0 , np.arange(arr.shape[1]), arr.shape[1] - 1)
    idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1].copy()

    out = arr[np.arange(idx.shape[0])[:,None], idx].transpose(1,0)
    out = np.transpose(out.reshape(new_shape) , new_axes)
    return out

def index_intersect(idxs , min_value = None , max_value = None):
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.intersect1d(new_idx , idx)
    if min_value is not None: new_idx = new_idx[new_idx >= min_value]
    if max_value is not None: new_idx = new_idx[new_idx <= max_value]
    new_idx = np.sort(new_idx)
    inter   = [np.array([]) if idx is None else np.intersect1d(new_idx , idx , return_indices=True) for idx in idxs]
    pos_new = tuple(np.array([]) if v is None else v[1] for v in inter)
    pos_old = tuple(np.array([]) if v is None else v[2] for v in inter)
    return new_idx , pos_new , pos_old

def index_union(idxs , min_value = None , max_value = None):
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.union1d(new_idx , idx)
    if min_value is not None: new_idx = new_idx[new_idx >= min_value]
    if max_value is not None: new_idx = new_idx[new_idx <= max_value]
    inter   = [np.array([]) if idx is None else np.intersect1d(new_idx , idx , return_indices=True) for idx in idxs]
    pos_new = tuple(np.array([]) if v is None else v[1] for v in inter)
    pos_old = tuple(np.array([]) if v is None else v[2] for v in inter)
    return new_idx , pos_new , pos_old

def ask_for_confirmation(prompt ='' , timeout = 10 , recurrent = 1 , proceed_condition = lambda x:True , print_function = print):
    assert isinstance(prompt , str)
    userText_list , userText_cond = [] , []
    for t in range(recurrent):
        if t == 0:
            _prompt = prompt 
        elif t == 1:
            _prompt = 'Really?'
        else:
            _prompt = 'Really again?'
            
        userText, timedOut = None , None
        if timeout > 0:
            try:
                userText, timedOut = timedInput(f'{_prompt} (in {timeout} seconds): ' , timeout = timeout)
            except:
                ...
        if userText is None : 
            userText, timedOut = input(f'{_prompt} : ') , False
        (_timeout , _sofar) = ('Time Out! ' , 'so far') if timedOut else ('' , '')
        print_function(f'{_timeout}User-input {_sofar} is : [{userText}].')
        userText_list.append(userText)
        userText_cond.append(proceed_condition(userText))
        if not userText_cond[-1]: break
    return userText_list , userText_cond

def recur_update(old : dict , update : Optional[dict]) -> dict:
    if update:
        for k , v in update.items():
            if isinstance(v , dict) and isinstance(old.get(k) , dict):
                old[k] = recur_update(old[k] , v)
            else:
                old[k] = v
    return old

class Filtered:
    def __init__(self, iterable, condition , **kwargs):
        self.iterable  = iter(iterable)
        self.condition = condition if callable(condition) else iter(condition)
        self.kwargs = kwargs
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: return item
