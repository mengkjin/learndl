from src.res.model.util import TrainConfig

from .nn import NNPredictor
from .boost import BoostPredictor
from .nn_booster import NNBooster
from .db import DBPredictor

def get_predictor_module(module : str | TrainConfig , *args , **kwargs):
    if isinstance(module , str):
        module = TrainConfig.default(module)
    module_type = module.module_type
    booster_head = module.model_booster_head

    if module_type == 'nn':
        mod = NNPredictor if not booster_head else NNBooster
    elif module_type == 'booster':
        mod = BoostPredictor
    elif module_type == 'db':
        mod = DBPredictor
    predictor = mod(*args , **kwargs)
    predictor.bound_with_config(module)
    return predictor
    