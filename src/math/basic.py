import torch , sys
import numpy as np
from typing import Any

DIV_TOL = 1e-6

def alert_message(message : str , color : str = 'lightyellow'):
    if color == 'lightyellow':
        sys.stderr.write(f'\u001b[93m\u001b[1m{message}\u001b[0m')
    elif color == 'lightred':
        sys.stderr.write(f'\u001b[91m\u001b[1m{message}\u001b[0m')
    else:
        sys.stderr.write(message)

def same(x , y):
    if type(x) is not type(y):
        return False
    elif isinstance(x , torch.Tensor):
        return x.equal(y)
    else:
        return (x == y).all()

def allna(x : torch.Tensor | np.ndarray | None , inf_as_na = True):
    if x is None:
        return True
    elif inf_as_na:
        return not x.isfinite().any() if isinstance(x , torch.Tensor) else not np.isfinite(x).any()
    else:
        return x.isnan().all() if isinstance(x , torch.Tensor) else not np.isnan(x).all()

def exact(x , y):
    return x is y

def average_params(params_list : tuple[dict] | list[dict]):
    n = len(params_list)
    if n == 1: 
        return params_list[0]
    new_params = {k:v / n for k,v in params_list[0].items()}
    for i, params in enumerate(params_list[1:]):
        for k, v in params.items():
            if k not in new_params.keys(): 
                raise KeyError(f'the {i}-th model has different params named {k}')
            else:
                new_params[k] += v / n
    return new_params

def to_numpy(values):
    """convert values to numpy array"""
    if not isinstance(values , np.ndarray): 
        if isinstance(values , torch.Tensor): 
            values = values.cpu().numpy()
        elif isinstance(values , list):
            values = np.asarray(values)
        else: 
            values = np.asarray(values).reshape(-1)
    return values

def match_values(values , src_arr , ambiguous = 0):
    values = to_numpy(values)
    src_arr = to_numpy(src_arr)
    sorter = np.argsort(src_arr)
    index = np.tile(len(src_arr) , values.shape)
    if ambiguous == 0:
        index[np.isin(values , src_arr)] = sorter[np.searchsorted(src_arr, values[np.isin(values , src_arr)], sorter=sorter)]
    else:
        index[values <= max(src_arr)] = sorter[np.searchsorted(src_arr, values[values <= max(src_arr)], sorter=sorter)]
    return index
    
def forward_fillna(arr , axis = 0):
    shape = arr.shape
    if axis < 0: 
        axis = len(shape) + axis
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
    if axis < 0: 
        axis = len(shape) + axis
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
    new_idx : np.ndarray | Any = None
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.intersect1d(new_idx , idx)
    if min_value is not None: 
        new_idx = new_idx[new_idx >= min_value]
    if max_value is not None: 
        new_idx = new_idx[new_idx <= max_value]
    new_idx = np.sort(new_idx)
    inter   = [np.array([]) if idx is None else np.intersect1d(new_idx , idx , return_indices=True) for idx in idxs]
    pos_new = tuple(np.array([]) if v is None else v[1] for v in inter)
    pos_old = tuple(np.array([]) if v is None else v[2] for v in inter)
    return new_idx , pos_new , pos_old

def index_union(idxs , min_value = None , max_value = None) -> tuple[np.ndarray , tuple[np.ndarray , ...] , tuple[np.ndarray , ...]]:
    new_idx : np.ndarray | Any = None
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.union1d(new_idx , idx)
    if min_value is not None: 
        new_idx = new_idx[new_idx >= min_value]
    if max_value is not None: 
        new_idx = new_idx[new_idx <= max_value]
    inter   = [np.array([]) if idx is None else np.intersect1d(new_idx , idx , return_indices=True) for idx in idxs]
    pos_new = tuple(np.array([]) if v is None else v[1] for v in inter)
    pos_old = tuple(np.array([]) if v is None else v[2] for v in inter)
    return new_idx , pos_new , pos_old

def index_stack(idxs , min_value = None , max_value = None) -> tuple[np.ndarray , tuple[np.ndarray , ...] , tuple[np.ndarray , ...]]:
    new_idx : np.ndarray | Any = None
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        elif np.isin(idx , new_idx).all():
            ...
        elif not np.isin(new_idx , idx).any():
            new_idx = np.concatenate([new_idx , idx])
        else:
            raise ValueError(f'idx {idx} is not a subset of new_idx {new_idx} , index_stack inputs must be identical or completely distinct index')
    if min_value is not None: 
        new_idx = new_idx[new_idx >= min_value]
    if max_value is not None: 
        new_idx = new_idx[new_idx <= max_value]
    inter   = [np.array([]) if idx is None else np.intersect1d(new_idx , idx , return_indices=True) for idx in idxs]
    pos_new = tuple(np.array([]) if v is None else v[1] for v in inter)
    pos_old = tuple(np.array([]) if v is None else v[2] for v in inter)
    return new_idx , pos_new , pos_old