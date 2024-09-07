import torch
import numpy as np

from torch import Tensor , nn
from typing import Any , Iterator , Optional

from ...util import BatchData , BatchOutput
from ....algo.boost import BoosterInput

def tensor_refiller(values : Optional[Tensor] , target_i0 , target_i1 , target_shape : tuple):
    if values is None: return None
    assert len(target_shape) in [2,3] , target_shape
    new_values = torch.full(target_shape , fill_value = np.nan)
    if len(target_shape) == 2 and values.ndim == 2:
        new_values[target_i0 , target_i1] = values[:,0]
    else:
        new_values[target_i0 , target_i1] = values[:]
    return new_values

def batch_data_flatten_x(batch_data : BatchData):
    if isinstance(batch_data.x , Tensor):
        x = batch_data.x.flatten(1)
    else:
        x = torch.concat([x.flatten(1) for x in batch_data.x] , -1)
    return x

def batch_data_to_boost_input(batch_data : BatchData , 
                              secid : Optional[np.ndarray] = None ,
                              date : Optional[np.ndarray] = None , 
                              nn_to_calculate_hidden : Optional[nn.Module] = None):
    if nn_to_calculate_hidden is not None:
        hidden : Tensor = BatchOutput(nn_to_calculate_hidden(batch_data.x)).other['hidden']
        assert hidden is not None , f'hidden must not be none when using BoosterModel'
        xx = hidden.detach().cpu()
    else:
        xx = batch_data_flatten_x(batch_data)
    ii = batch_data.i.cpu()

    secid_i , secid_j = np.unique(ii[:,0].numpy() , return_inverse=True)
    date_i  , date_j  = np.unique(ii[:,1].numpy() , return_inverse=True)

    xx_values = tensor_refiller(xx , secid_j , date_j , (len(secid_i) , len(date_i) , xx.shape[-1]))
    yy_values = tensor_refiller(batch_data.y , secid_j , date_j , (len(secid_i) , len(date_i)))
    ww_values = tensor_refiller(batch_data.w , secid_j , date_j , (len(secid_i) , len(date_i)))

    assert xx_values is not None
    secid = secid[secid_i] if secid is not None else None
    date  = date[date_i]   if date  is not None else None

    return BoosterInput.from_tensor(xx_values , yy_values , ww_values , secid , date)

def batch_loader_concat(batch_loader : Iterator[BatchData] , nn_to_calculate_hidden : Optional[nn.Module] = None):
    xx , yy , ww , ii , vv = [] , [] , [] , [] , []
    for batch_data in batch_loader:
        if nn_to_calculate_hidden is not None:
            hidden : Tensor = BatchOutput(nn_to_calculate_hidden(batch_data.x)).other['hidden']
            assert hidden is not None , f'hidden must not be none when using BoosterModel'
            xx.append(hidden.detach().cpu())
        batch_data = batch_data.cpu()
        if nn_to_calculate_hidden is None:
            xx.append(batch_data.x)

        ww.append(batch_data.w)
        yy.append(batch_data.y)
        ii.append(batch_data.i)
        vv.append(batch_data.valid)
    return BatchData(comp_cat(xx) , comp_cat(yy) , comp_cat(ww) , comp_cat(ii) , comp_cat(vv))

def comp_cat(comp_list : list) -> Any:
    assert all([isinstance(x , type(comp_list[0])) for x in comp_list]) , comp_list
    if comp_list[0] is None:
        return None
    elif isinstance(comp_list[0] , Tensor):
        return torch.concat(comp_list)
    else:
        assert all([len(x) == len(comp_list[0]) for x in comp_list]) , comp_list
        return type(comp_list[0])([torch.concat([x[i] for x in comp_list]) for i in range(len(comp_list[0]))])


def batch_loader_to_boost_input(batch_loader : Iterator[BatchData] , 
                                secid : Optional[np.ndarray] = None ,
                                date : Optional[np.ndarray] = None , 
                                nn_to_calculate_hidden : Optional[nn.Module] = None):
    large_batch = batch_loader_concat(batch_loader , nn_to_calculate_hidden)
    return batch_data_to_boost_input(large_batch , secid , date)