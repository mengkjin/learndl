import torch

from torch import Tensor
from typing import Any , Optional , Literal

from .nn.api import valid_nn , get_nn_module , get_multiloss_params , get_nn_category , get_nn_datatype , AVAILABLE_NNS
from .boost.api import valid_booster , AVAILABLE_BOOSTERS , OptunaBooster , GeneralBooster

def module_type(module_name : str):
    if module_name in ['booster' , 'hidden_aggregator'] or valid_booster(module_name): return 'boost'
    if valid_nn(module_name): return 'nn'
    raise ValueError(f'{module_name} is not a valid module')

def nn(model_module : str , model_param : dict | None = None , device : Any = None , state_dict : Optional[dict[str,Tensor]] = None , **kwargs):
    model_param = model_param or {}
    net = get_nn_module(model_module)(**model_param)
    assert isinstance(net , torch.nn.Module) , net.__class__
    if state_dict: net.load_state_dict(state_dict)
    net.eval()
    return device(net) if callable(device) else net.to(device)

def boost(model_module : str , model_param : dict | None = None , cuda = None , seed = None , model_dict : Optional[dict] = None , 
          given_name : Optional[str] = None , optuna : bool = False , **kwargs):
    model_param = model_param or {}
    booster = (OptunaBooster if optuna else GeneralBooster)(
        model_module , model_param , cuda = bool(cuda) , seed = seed , given_name = given_name , **kwargs)

    if model_dict is not None: booster.load_dict(model_dict , cuda = bool(cuda) , seed = seed)
    return booster

def multiloss_params(module : torch.nn.Module | Any): return get_multiloss_params(module)

def nn_category(module_name : str): return get_nn_category(module_name)

def nn_datatype(module_name : str): return get_nn_datatype(module_name)

def available_modules(module_type : Literal['nn' , 'boost' , 'all'] = 'all'):
    if module_type == 'all':
        return {**AVAILABLE_NNS , **AVAILABLE_BOOSTERS}
    elif module_type == 'nn':
        return AVAILABLE_NNS
    elif module_type == 'boost':
        return AVAILABLE_BOOSTERS
    else:
        raise ValueError(f'{module_type} is not a valid module type')