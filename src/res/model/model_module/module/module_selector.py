from src.res.model.util import ModelConfig
from src.res.model.util.abc import is_null_module_type

from .nn import NNPredictor
from .boost import BoostPredictor
from .nn_boost import NNBoost
from .null import NullPredictor

def get_predictor_module(module : str | ModelConfig , *args , **kwargs):
    if isinstance(module , str):
        module = ModelConfig.default(module = module)
    module_type = module.module_type
    boost_head = module.boost_head

    if module_type == 'nn':
        mod = NNPredictor if not boost_head else NNBoost
    elif module_type == 'boost':
        mod = BoostPredictor
    elif is_null_module_type(module_type):
        mod = NullPredictor
    else:
        raise ValueError(f'invalid module type: {module_type}')
    predictor = mod(*args , **kwargs).bound_with(module)
    return predictor
    