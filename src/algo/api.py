import torch

from torch import Tensor
from typing import Any , Optional , Literal

from src.project_setting import MACHINE
from .nn.api import get_nn_module , get_multiloss_params , get_nn_category , get_nn_datatype , AVAILABLE_NNS
from .boost.api import AVAILABLE_BOOSTERS , OptunaBooster , GeneralBooster

class AlgoModule:
    AVAILABLE_BOOSTERS = AVAILABLE_BOOSTERS
    AVAILABLE_NNS      = AVAILABLE_NNS
    AVAILABLE_MODULES  = {**AVAILABLE_NNS , **AVAILABLE_BOOSTERS}
    
    @classmethod
    def available_modules(cls , module_type : Literal['nn' , 'boost' , 'all'] = 'all'):
        if module_type == 'all':
            return cls.AVAILABLE_MODULES
        elif module_type == 'nn':
            return cls.AVAILABLE_NNS
        elif module_type == 'boost':
            return cls.AVAILABLE_BOOSTERS
        else:
            raise ValueError(f'{module_type} is not a valid module type')
        
    @classmethod
    def available_modules_str(cls , module_type : Literal['nn' , 'boost' , 'all'] = 'all'):
        if module_type == 'all':
            return '\n'.join([cls.available_modules_str('nn') , cls.available_modules_str('boost')])
        elif module_type == 'nn':
            return '\n'.join([f'nn/{module}' for module in cls.AVAILABLE_NNS.keys()])
        elif module_type == 'boost':
            return '\n'.join([f'boost/{module}' for module in cls.AVAILABLE_BOOSTERS.keys()])
        else:
            raise ValueError(f'{module_type} is not a valid module type')
        
    @classmethod
    def export_available_modules(cls):
        with open(f'{MACHINE.project_path}/available_modules.txt' , 'w') as f:
            f.write(cls.available_modules_str())
        
    @classmethod
    def is_valid(cls , model_module : str , type : Literal['nn' , 'boost' , 'all'] = 'all'): 
        if type == 'all': return model_module in cls.AVAILABLE_MODULES
        elif type == 'nn': return model_module in cls.AVAILABLE_NNS
        elif type == 'boost': return model_module in cls.AVAILABLE_BOOSTERS
        else: raise ValueError(f'{type} is not a valid boost type')

    @classmethod
    def module_type(cls , module_name : str):
        if module_name in ['booster' , 'hidden_aggregator'] or module_name in cls.AVAILABLE_BOOSTERS: return 'boost'
        if module_name in cls.AVAILABLE_NNS: return 'nn'
        raise ValueError(f'{module_name} is not a valid module')
    
    @classmethod
    def get_nn(cls , model_module : str , model_param : dict | None = None , device : Any = None , state_dict : Optional[dict[str,Tensor]] = None , **kwargs):
        model_param = model_param or {}
        net = get_nn_module(model_module)(**model_param)
        assert isinstance(net , torch.nn.Module) , net.__class__
        if state_dict: net.load_state_dict(state_dict)
        net.eval()
        return device(net) if callable(device) else net.to(device)
    
    @classmethod
    def nn_category(cls , module_name : str): return get_nn_category(module_name)
    
    @classmethod
    def nn_datatype(cls , module_name : str): return get_nn_datatype(module_name)

    
    @classmethod
    def get_booster(cls , model_module : str , model_param : dict | None = None , cuda = None , seed = None , model_dict : Optional[dict] = None , 
          given_name : Optional[str] = None , optuna : bool = False , **kwargs):
        model_param = model_param or {}
        booster = (OptunaBooster if optuna else GeneralBooster)(
            model_module , model_param , cuda = bool(cuda) , seed = seed , given_name = given_name , **kwargs)

        if model_dict is not None: booster.load_dict(model_dict , cuda = bool(cuda) , seed = seed)
        return booster

    @classmethod
    def multiloss_params(cls , module : torch.nn.Module | Any): return get_multiloss_params(module)

AlgoModule.export_available_modules()