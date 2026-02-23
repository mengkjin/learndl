import torch
import torch.nn as nn
import numpy as np

from typing import Iterator

from src.res.algo.boost import BoosterInput
from src.res.model.util import BatchInput , BatchOutput

def tensor_refiller(values : torch.Tensor | None , target_i0 , target_i1 , target_shape : tuple):
    if values is None: 
        return None
    assert len(target_shape) in [2,3] , target_shape
    new_values = torch.full(target_shape , fill_value = np.nan)
    if len(target_shape) == 2 and values.ndim == 2:
        new_values[target_i0 , target_i1] = values[:,0]
    else:
        new_values[target_i0 , target_i1] = values[:]
    return new_values

def batch_data_flatten_x(batch_input : BatchInput):
    if isinstance(batch_input.x , torch.Tensor):
        x = batch_input.x.flatten(1)
    else:
        x = torch.concat([x.flatten(1) for x in batch_input.x] , -1)
    return x

def batch_data_to_boost_input(batch_input : BatchInput , 
                              secid : np.ndarray | None = None ,
                              date : np.ndarray | None = None , 
                              nn_to_calculate_hidden : nn.Module | None = None):
    if nn_to_calculate_hidden is not None:
        hidden : torch.Tensor = BatchOutput(nn_to_calculate_hidden(batch_input.x)).other['hidden']
        assert hidden is not None , f'hidden must not be none when using BoosterModel'
        xx = hidden.detach().cpu()
    else:
        xx = batch_data_flatten_x(batch_input)
    ii = batch_input.i.cpu()

    secid_i , secid_j = np.unique(ii[:,0].numpy() , return_inverse=True)
    date_i  , date_j  = np.unique(ii[:,1].numpy() , return_inverse=True)

    xx_values = tensor_refiller(xx , secid_j , date_j , (len(secid_i) , len(date_i) , xx.shape[-1]))
    yy_values = tensor_refiller(batch_input.y , secid_j , date_j , (len(secid_i) , len(date_i)))
    ww_values = tensor_refiller(batch_input.w , secid_j , date_j , (len(secid_i) , len(date_i)))

    assert xx_values is not None , f'xx_values must not be none'
    secid = secid[secid_i] if secid is not None else None
    date  = date[date_i]   if date  is not None else None

    return BoosterInput.from_tensor(xx_values , yy_values , ww_values , secid , date)

def batch_x(batch_input : BatchInput , nn_to_calculate_hidden : nn.Module | None = None):
    if nn_to_calculate_hidden is not None:
        nn_to_calculate_hidden.eval()
        with torch.no_grad():
            hidden : torch.Tensor = BatchOutput(nn_to_calculate_hidden(batch_input.x)).other['hidden']
            assert hidden is not None , f'hidden must not be none when using BoosterModel'
        return hidden.detach()
    else:
        return batch_input.x
    
def batch_loader_concat(batch_loader : Iterator[BatchInput] , nn_to_calculate_hidden : nn.Module | None = None):
    new_batchs : list[BatchInput] = []
    for b in batch_loader:
        new_batch_data = BatchInput(batch_x(b , nn_to_calculate_hidden) , b.y , b.w , b.i , b.valid)
        new_batchs.append(new_batch_data.cpu())
    return BatchInput.concat(*new_batchs)

def batch_loader_to_boost_input(batch_loader : Iterator[BatchInput] , 
                                secid : np.ndarray | None = None ,
                                date : np.ndarray | None = None , 
                                nn_to_calculate_hidden : nn.Module | None = None):
    large_batch = batch_loader_concat(batch_loader , nn_to_calculate_hidden)
    return batch_data_to_boost_input(large_batch , secid , date)