import torch , sys
import numpy as np
from typing import Any , Literal

DIV_TOL = 1e-6

def alert_message(message : str , color : str = 'lightyellow'):
    if color == 'lightyellow':
        sys.stderr.write(f'\u001b[93m\u001b[1m{message}\u001b[0m')
    elif color == 'lightred':
        sys.stderr.write(f'\u001b[91m\u001b[1m{message}\u001b[0m')
    else:
        sys.stderr.write(message)

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
    """match values to src_arr , return the index of values in src_arr , no match return len(src_arr)"""
    values = to_numpy(values)
    src_arr = to_numpy(src_arr)
    sorter = np.argsort(src_arr)
    index = np.tile(len(src_arr) , values.shape)
    if ambiguous == 0:
        index[np.isin(values , src_arr)] = sorter[np.searchsorted(src_arr, values[np.isin(values , src_arr)], sorter=sorter)]
    else:
        index[values <= max(src_arr)] = sorter[np.searchsorted(src_arr, values[values <= max(src_arr)], sorter=sorter)]
    return index

def convert_to_slice(index : np.ndarray | list) -> slice | np.ndarray:
    if len(index) == 0:
        return slice(0,0,1)
    elif len(index) == 1:
        return slice(index[0],index[0]+1,1)
    else:
        start = min(index)
        end = max(index) + 1
        step = index[1] - index[0]
        if np.array_equal(index , np.arange(start,end,step)):
            return slice(start,end,step)
        else:
            return to_numpy(index)

def match_slice(values , src_arr , ambiguous = 0) -> slice | np.ndarray:
    if len(values) == 0:
        return slice(0,0,1)
    elif np.array_equal(values , src_arr):
        return slice(None,None,1)
    else:
        return convert_to_slice(match_values(values , src_arr , ambiguous))

def array_to_tensor_slice(array : np.ndarray | list) -> slice | torch.Tensor:
    array_slice = convert_to_slice(array)
    if isinstance(array_slice , slice):
        return array_slice
    else:
        return torch.from_numpy(array_slice)
    
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

def trim_index(index : np.ndarray | Any , min_value = None , max_value = None) -> np.ndarray:
    if min_value is not None:
        index = index[index >= min_value]
    if max_value is not None:
        index = index[index <= max_value]
    return index

def index_intersect(idxs , min_value = None , max_value = None) -> np.ndarray:
    """intersect multiple index arrays , return the intersect index , the position in the target array , the position in the original index"""
    new_idx : np.ndarray | torch.Tensor | Any = None
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.intersect1d(new_idx , idx)
    new_idx = np.sort(new_idx)
    return trim_index(new_idx , min_value , max_value)

def index_union(idxs , min_value = None , max_value = None) -> np.ndarray:
    """union multiple index arrays , return the union index , the position in the target array , the position in the original index"""
    new_idx : np.ndarray | Any = None
    for i , idx in enumerate(idxs):
        if i == 0 or idx is None or new_idx is None:
            new_idx = new_idx if idx is None else idx
        else:
            new_idx = np.union1d(new_idx , idx)
    return trim_index(new_idx , min_value , max_value)

def index_stack(idxs , min_value = None , max_value = None) -> np.ndarray:
    """stack multiple index arrays , return the stack index , the position in the target array , the position in the original index"""
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
    return trim_index(new_idx , min_value , max_value)

def index_check(idxs , min_value = None , max_value = None) -> np.ndarray:
    """stack multiple index arrays , return the stack index , the position in the target array , the position in the original index"""
    new_idx : np.ndarray | Any = None
    for i , idx in enumerate(idxs):
        if i == 0:
            new_idx = idx
        elif np.array_equal(idx , new_idx):
            pass
        else:
            raise ValueError(f'idx at {i} is {idx}, does not equal to idx at 0 {new_idx} , index_check inputs must be identical')
    return trim_index(new_idx , min_value , max_value)

def index_merge(idxs , * , method : Literal['intersect' , 'union' , 'stack' , 'check'] = 'intersect' , 
               min_value = None , max_value = None) -> np.ndarray:
    assert len(idxs) > 0 , 'index_check inputs must be a non-empty list'
    if method == 'intersect':
        return index_intersect(idxs , min_value , max_value)
    elif method == 'union':
        return index_union(idxs , min_value , max_value)
    elif method == 'stack':
        return index_stack(idxs , min_value , max_value)
    elif method == 'check':
        return index_check(idxs , min_value , max_value)
    else:
        raise ValueError(f'Invalid method: {method}')

def intersect_pos_tensor(target_index : np.ndarray | Any , source_index : np.ndarray | Any) -> tuple[torch.Tensor , torch.Tensor]:
    _ , target_pos , source_pos = np.intersect1d(target_index , source_index , return_indices=True)
    return torch.from_numpy(target_pos), torch.from_numpy(source_pos)

def intersect_pos_slice(target_index : np.ndarray | Any , source_index : np.ndarray | Any) -> tuple[slice | np.ndarray , slice | np.ndarray]:
    _ , target_pos , source_pos = np.intersect1d(target_index , source_index , return_indices=True)
    return convert_to_slice(target_pos), convert_to_slice(source_pos)

def intersect_meshgrid(target_indices : list[torch.Tensor | np.ndarray | Any] , source_indices : list[torch.Tensor | np.ndarray | Any]) -> tuple[tuple[torch.Tensor , ...] , tuple[torch.Tensor , ...]]:
    assert len(target_indices) == len(source_indices) , f'target_indices and source_indices must have the same length , but got {len(target_indices)} and {len(source_indices)}'
    target_pos = []
    source_pos = []
    for target_index , source_index in zip(target_indices, source_indices):
        tarpos , srcpos = intersect_pos_tensor(target_index , source_index)
        target_pos.append(tarpos)
        source_pos.append(srcpos)
    return torch.meshgrid(*target_pos, indexing='ij'), torch.meshgrid(*source_pos, indexing='ij')


