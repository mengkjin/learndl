import torch

from torch import Tensor
from typing import Any , Optional

from .nn import api
from .boost import GeneralBooster , VALID_BOOSTERS

def nn(model_module : str , model_param = {} , device : Any = None , state_dict : Optional[dict[str,Tensor]] = {} , **kwargs):
    net = api.get_nn_module(model_module)(**model_param)
    assert isinstance(net , torch.nn.Module) , net.__class__
    if state_dict: net.load_state_dict(state_dict)
    net.eval()
    return device(net) if callable(device) else net.to(device)

def boost(model_module : str , model_param = {} , cuda = None , seed = None , model_dict : Optional[dict] = None , given_name : Optional[str] = None):
    assert model_module in VALID_BOOSTERS, model_module
    booster = GeneralBooster(model_module , model_param , cuda = bool(cuda) , seed = seed , given_name = given_name)
    if model_dict is not None: booster.load_dict(model_dict , cuda = bool(cuda) , seed = seed)
    return booster

def multiloss_params(module : torch.nn.Module | Any): return api.get_multiloss_params(module)

def nn_category(module_name : str): return api.get_nn_category(module_name)

def nn_datatype(module_name : str): return api.get_nn_datatype(module_name)