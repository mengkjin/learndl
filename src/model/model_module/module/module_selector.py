from .nn import NNPredictor
from .boost import BoostPredictor
from .nn_booster import NNBooster
from ...util import TrainConfig

def get_predictor_module(model_module : str | TrainConfig , *args , **kwargs):
    if isinstance(model_module , str):
        module_type = TrainConfig.get_module_type(model_module)
        if module_type == 'nn':
            mod = NNPredictor
        else:
            mod = BoostPredictor
        return mod(*args , **kwargs)
    elif isinstance(model_module , TrainConfig):
        config = model_module
        if config.module_type == 'nn' and config.model_booster_head:
            mod = NNBooster
        elif config.module_type == 'nn':
            mod = NNPredictor
        else:
            mod = BoostPredictor
        return mod(*args , **kwargs).bound_with_config(config)