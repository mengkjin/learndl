"""
select the appropriate predictor module based on input module name or ModelConfig
"""

from __future__ import annotations
from src.res.model.util import ModelConfig , is_null_module_type , PredictorModel

__all__ = ['get_predictor_module']

def get_predictor_module(module : str | ModelConfig , *args , **kwargs) -> PredictorModel:
    if isinstance(module , str):
        module = ModelConfig(module = module)
    module_type = module.module_type
    boost_head = module.boost_head

    if module_type == 'nn' and boost_head:
        from src.res.model.model_module.module.nn_boost import NNBoost
        predictor = NNBoost(*args , **kwargs)
    elif module_type == 'nn':
        from src.res.model.model_module.module.nn import NNPredictor
        predictor = NNPredictor(*args , **kwargs)
    elif module_type == 'boost':
        from src.res.model.model_module.module.boost import BoostPredictor
        predictor = BoostPredictor(*args , **kwargs)
    elif is_null_module_type(module_type):
        from src.res.model.model_module.module.null import NullPredictor
        predictor = NullPredictor(*args , **kwargs)
    else:
        raise ValueError(f'invalid module type: {module_type}')
    predictor.bound_with(module)
    return predictor    