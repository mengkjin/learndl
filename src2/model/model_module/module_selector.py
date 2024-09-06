from .nn_module.module import NNPredictor
from .boost_module.module import BoostPredictor
from ..util import TrainConfig

def module_selector(model_module : str):
    module_type = TrainConfig.get_module_type(model_module)
    if module_type == 'nn':
        return NNPredictor
    else:
        return BoostPredictor