"""Top-level registry and factory for all algorithm modules (NN + boost).

Classes:
    AlgoModule — class-method registry that can instantiate any model by name,
                 resolve its type (``'nn'`` or ``'boost'``), and export the full
                 module list to a text file.

.. warning::
    ``AlgoModule.export_available_modules()`` is called at module import time,
    writing a file to ``PATH.temp``.  This is a side-effect that should be
    made opt-in (see ``TODO_res_algo.md``).
"""
import torch

from typing import Any , Literal

from src.proj import PATH
from .nn.api import get_nn_module , get_nn_category , get_nn_datatype , AVAILABLE_NNS
from .boost.api import AVAILABLE_BOOSTS , OptunaBoostModel , GeneralBoostModel

__all__ = ['AlgoModule' , 'get_nn_module' , 'get_nn_category' , 'get_nn_datatype' , 'OptunaBoostModel' , 'GeneralBoostModel']

class AlgoModule:
    """Unified registry and factory for all NN and boost models.

    Class attributes:
        _availables: ``{'nn': AVAILABLE_NNS, 'boost': AVAILABLE_BOOSTS}`` —
                     nested dict of ``{name: class}`` mappings.

    Type heuristics in :meth:`module_type`:
        * ``'boost'`` — name is ``'boost'``, in ``AVAILABLE_BOOSTS``, or
          starts with ``'boost@'``.
        * ``'nn'``    — name is ``'nn'``, in ``AVAILABLE_NNS``, or starts
          with ``'nn@'``.
    """
    _availables = {'boost' : AVAILABLE_BOOSTS , 'nn' : AVAILABLE_NNS}
    
    @classmethod
    def available_modules(cls , module_type : Literal['nn' , 'boost' , 'all'] = 'all'):
        if module_type == 'all':
            return {k:v for mod_type, mods in cls._availables.items() for k,v in mods.items()}
        else:
            return cls._availables[module_type]
        
    @classmethod
    def available_modules_str(cls , module_type : Literal['nn' , 'boost' , 'all'] = 'all'):
        if module_type == 'all':
            return '\n'.join([cls.available_modules_str('nn') , cls.available_modules_str('boost')])
        else:
            return '\n'.join([f'{module_type}/{module}' for module in cls._availables[module_type].keys()])
        
    @classmethod
    def export_available_modules(cls):
        path = PATH.temp.joinpath('available_modules.txt')
        path.parent.mkdir(parents = True , exist_ok = True)
        with open(path , 'w') as f:
            f.write(cls.available_modules_str())
        
    @classmethod
    def is_valid(cls , model_module : str , module_type : Literal['nn' , 'boost' , 'all'] = 'all'): 
        if module_type == 'all': 
            return cls.is_valid(model_module , 'nn') or cls.is_valid(model_module , 'boost')
        else: 
            return model_module in cls._availables[module_type]

    @classmethod
    def module_type(cls , module_name : str , raise_error : bool = True):
        if module_name == 'boost' or module_name in cls._availables['boost'] or module_name.startswith('boost@'): 
            return 'boost'
        if module_name == 'nn' or module_name in cls._availables['nn'] or module_name.startswith('nn@'): 
            return 'nn'
        if raise_error:
            raise ValueError(f'{module_name} is not a valid module')
        else:
            return ''
    
    @classmethod
    def get_nn(cls , model_module : str , model_param : dict | None = None , device : Any = None , state_dict : dict[str,torch.Tensor] | None = None , **kwargs):
        model_param = model_param or {}
        net : torch.nn.Module | Any = get_nn_module(model_module)(**model_param)
        assert isinstance(net , torch.nn.Module) , net.__class__
        if state_dict: 
            net.load_state_dict(state_dict)
        net.eval()
        net = device(net) if callable(device) else net.to(device)
        return net
    
    @classmethod
    def nn_category(cls , module_name : str): 
        return get_nn_category(module_name)
    
    @classmethod
    def nn_datatype(cls , module_name : str): 
        return get_nn_datatype(module_name)

    @classmethod
    def get_boost(
        cls , model_module : str , model_param : dict | None = None , 
        cuda = None , seed = None , model_dict : dict | None = None , 
        given_name : str | None = None , optuna : bool = False , **kwargs
    ):
        model_param = model_param or {}
        boost = (OptunaBoostModel if optuna else GeneralBoostModel)(
            model_module , model_param , cuda = bool(cuda) , seed = seed , given_name = given_name , **kwargs)

        if model_dict is not None: 
            boost.load_dict(model_dict , cuda = bool(cuda) , seed = seed)
        return boost

AlgoModule.export_available_modules()