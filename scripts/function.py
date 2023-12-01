import numpy as np
import pandas as pd
import torch
import time , os , shutil , pprint
from scipy import stats
from pytimedinput import timedInput

def emphasize_header(header=''):
    print('{: ^100}'.format(''))
    print('{:*^100}'.format(''))
    print('{:*^100}'.format('    '+header+'    '))
    print('{:*^100}'.format(''))
    print('{: ^100}'.format(''))
        
def tensor_nancount(x, dim=None, keepdim=False):  
    return (1-x.isnan().int()).sum(dim = dim , keepdim = keepdim)

def tensor_nanmean(x, dim=None, keepdim=False):  
    try:
        return x.nanmean(dim = dim , keepdim = keepdim)
    except:
        return x.nansum(dim = dim , keepdim = keepdim) / tensor_nancount(x , dim = dim , keepdim = keepdim)

def tensor_nanstd(x, dim=None, keepdim=False , correction=1):
    if dim is None:
        return torch.tensor(np.nanstd(x.flatten()))
    nancount = tensor_nancount(x , dim = dim , keepdim = True) - correction
    return ((x - tensor_nanmean(x , dim = dim , keepdim = True)).square() / nancount).nansum(dim = dim , keepdim = keepdim).sqrt()

def tensor_nanmedian(x, dim=None, keepdim=False):
    raw_dtype = x.dtype
    if raw_dtype in [torch.float , torch.double]:
        if dim is None:
            return x.nanmedian()
        else:
            return x.nanmedian(dim = dim , keepdim = keepdim).values
    else:
        if dim is None:
            return x.to(torch.float).nanmedian().to(raw_dtype)
        else:
            return x.to(torch.float).nanmedian(dim = dim , keepdim = keepdim).values.to(raw_dtype)
        
def tensor_standardize_and_weight(x, dim=None , weight_scheme = None):
    if x.isnan().all().item():
        return (x , None)       
    x = (x - tensor_nanmean(x,dim=dim,keepdim=True)) / (tensor_nanstd(x,dim=dim,correction=0,keepdim=True) + 1e-4)
    w = None
    if weight_scheme is None: 
        pass
    else:
        weight_scheme = 'equal' if weight_scheme not in ['top'] else weight_scheme
        if weight_scheme == 'top':
            w = torch.ones_like(x)
            try: 
                w[x > tensor_nanmedian(x , dim , True)] = 2
            except:    
                w[x > tensor_nanmedian(x)] = 2
    return (x, w)

def standardize_x(x , dim=None):
    if np.all(np.isnan(x)):
        pass
    elif dim is None or len(x.shape) == 1:
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-4)
    else:
        tran_dim = np.arange(len(x.shape))
        tran_dim[0],tran_dim[dim] = dim,0
        y = x.transpose(*tran_dim).reshape(x.shape[dim],-1) * 1.
        for i in range(y.shape[-1]):
            y[:,i] = standardize_x(y[:,i])
        x = y.reshape(*[x.shape[j] for j in tran_dim]).transpose(*tran_dim)
    return x

def standardize_and_weight(x , dim=None):
    if np.all(np.isnan(x)):
        pass
    elif dim is None or len(x.shape) == 1:
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-4)
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

def multi_bin_label(x , n = 10):
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


def bin_label(x):
    y , w = np.zeros_like(x) , np.zeros_like(x)
    y[x >= np.nanmedian(x)] = 1
    w[:] = y + 1
    return y, w

def tensor_rank(x):    
    assert x.dim() == 1 , x.dim()
    return torch.zeros_like(x).index_copy_(0,x.argsort(),torch.arange(0.,len(x)))
def rank_weight(x):    
    r = tensor_rank(x)
    w = torch.pow(0.5,((r.numel() - 1 - r) * 2 / (r.numel() - 1)))
    return w / w.sum()
def nd_rank(x , dim = None):
    if dim is None:
        w = tensor_rank(x.flatten()).reshape(x.shape)
    else:
        w = torch.zeros_like(x).copy_(x).transpose(-1 , dim)
        new_shape = w.shape
        w = w.reshape(-1 , new_shape[-1])
        for i in range(len(w)):
            w[i] = tensor_rank(w[i])
        w = w.reshape(*new_shape).transpose(-1 , dim)   
    return w
def nd_rank_weight(x , dim = None):
    if dim is None:
        w = rank_weight(x.flatten()).reshape(x.shape)
    else:
        w = torch.zeros_like(x).copy_(x).transpose(-1 , dim)
        new_shape = w.shape
        w = w.reshape(-1 , new_shape[-1])
        for i in range(len(w)):
            w[i] = rank_weight(w[i])
        w = w.reshape(*new_shape).transpose(-1 , dim)   
    return w 
def nd_minus_mean(x , w , dim = None):
    return x - (w * x).mean(dim=dim,keepdim=True)

def pearson(x, y , w = None, dim = None , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    x1 , y1 = nd_minus_mean(x , w , dim) , nd_minus_mean(y , w , dim)
    return (w * x1 * y1).mean(dim = dim) / ((w * x1.square()).mean(dim=dim).sqrt() + 1e-4) / ((w * y1.square()).mean(dim=dim).sqrt() + 1e-4)
    
def ccc(x , y , w = None, dim = None , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    x1 , y1 = nd_minus_mean(x , w , dim) , nd_minus_mean(y , w , dim)
    cov_xy = (w * x1 * y1).mean(dim=dim)
    mse_xy = (w * (x1 - y1).square()).mean(dim=dim)
    return (2 * cov_xy) / (mse_xy + 2 * cov_xy + 1e-4)

def mse(x , y , w = None, dim = None , reduction='mean' , **kwargs):
    w = 1. if w is None else w / w.sum(dim=dim,keepdim=True) * (w.numel() if dim is None else w.size(dim=dim))
    f = torch.mean if reduction == 'mean' else torch.sum
    return f(w * (x - y).square() , dim=dim)

def spearman(x , y , w = None , dim = None , **kwargs):
    x , y = nd_rank(x , dim = dim) , nd_rank(y , dim = dim)
    return pearson(x , y , w , dim , **kwargs)

def wpearson(x, y , dim = None , **kwargs):
    w = nd_rank_weight(y , dim = dim)
    return pearson(x,y,w,dim)

def wccc(x , y , dim = None , **kwargs):
    w = nd_rank_weight(y , dim = dim)
    return ccc(x,y,w,dim)

def wmse(x , y , dim = None , reduction='mean' , **kwargs):
    w = nd_rank_weight(y , dim = dim)
    return mse(x,y,w,dim,reduction)

def wspearman(x , y , dim = None , **kwargs):
    w = nd_rank_weight(y , dim = dim)
    return spearman(x,y,w,dim,reduction)

def np_rankic(x , y , w = None , dim = None):
    return stats.spearmanr(x,y)[0]

def transpose_qkv(X,num_heads):
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])

def transpose_output(X,num_heads):
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)

def np_nanrankic(x , y):
    assert len(x) == len(y)
    pairwise_nonnan = (np.isnan(pxred)*1.0 + np.isnan(y) * 1.0 == 0)
    try:
        return np_rankic(x[pairwise_nonnan],y[pairwise_nonnan])
    except:
        return np.nan

def np_nanrankic_2d(x , y , dim = 0):
    assert type(x) == type(y)
    assert x.shape == y.shape
    if dim == 0:
        return [np_nanrankic(x[:,i],y[:,i]) for i in range(x.shape[1])]
    else:
        return [np_nanrankic(x[i,:],y[i,:]) for i in range(x.shape[0])]
    
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
                pass
        if userText is None : 
            userText, timedOut = input(f'{_prompt} : ') , False
        (_timeout , _sofar) = ('Time Out! ' , 'so far') if timedOut else ('' , '')
        print_function(f'{_timeout}User-input {_sofar} is : [{userText}].')
        userText_list.append(userText)
        userText_cond.append(proceed_condition(userText))
        if not userText_cond[-1]: 
            break
    return userText_list , userText_cond

def total_memory(unit = 1e9):
    return psutil.Process(os.getpid()).memory_info().rss / unit

def match_values(arr , values , ambiguous = 0):
    sorter = np.argsort(arr)
    index = np.tile(len(arr) , values.shape)
    if ambiguous == 0:
        index[np.isin(values , arr)] = sorter[np.searchsorted(arr, values[np.isin(values , arr)], sorter=sorter)]
    else:
        index[values <= max(arr)] = sorter[np.searchsorted(arr, values[values <= max(arr)], sorter=sorter)]
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

def rmdir(d , remake_dir = False):
    """
    Remove list/instance of dirs , and remake the dir if remake_dir = True
    """
    if isinstance(d , (list,tuple)):
        [shutil.rmtree(x) for x in d if os.path.exists(x)]
        if remake_dir : [os.makedirs(x , exist_ok = True) for x in d]
    elif isinstance(d , str):
        if os.path.exists(d): shutil.rmtree(d)
        if remake_dir : os.mkdir(d)
    else:
        raise Exception(f'KeyError : {str(d)}')
        
def list_converge(l , n = None , eps = None):
    """
    Last n element of l has range smaller than eps
    """
    n = len(l) if n is None else n
    eps = 0 if eps is None else eps
    return len(l) >= n and (max(l[-n:]) - min(l[-n:])) < eps   

def pretty_print_dict(dictionary , width = 140 , sort_dicts = False):
    pprint.pprint(dictionary, indent = 1, width = width , sort_dicts = sort_dicts)