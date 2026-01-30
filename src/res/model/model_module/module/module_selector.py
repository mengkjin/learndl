from src.res.model.util import TrainConfig

from .nn import NNPredictor
from .boost import BoostPredictor
from .nn_booster import NNBooster
from .null import NullPredictor

def get_predictor_module(module : str | TrainConfig , *args , **kwargs):
    if isinstance(module , str):
        module = TrainConfig.default(module)
    module_type = module.module_type
    booster_head = module.model_booster_head

    if module_type == 'nn':
        mod = NNPredictor if not booster_head else NNBooster
    elif module_type == 'booster':
        mod = BoostPredictor
    elif module_type in ['db' , 'factor']:
        mod = NullPredictor
    else:
        raise ValueError(f'invalid module type: {module_type}')
    predictor = mod(*args , **kwargs).bound_with(module)
    return predictor
    