import logging
from inspect import currentframe

from ..util.classes import BaseCB , TrainerStatus 
from ..util import Metrics , TrainConfig

class BasicCallBack(BaseCB):
    def __init__(self , model_module , *args , **kwargs) -> None:
        super().__init__(model_module)
    @property
    def config(self) -> TrainConfig:    return self.module.config
    @property
    def logger(self) -> logging.Logger: return self.module.logger
    @property
    def status(self) -> TrainerStatus:  return self.module.status
    @property
    def metrics(self) -> Metrics :  return self.module.metrics
            
class WithCallBack(BasicCallBack):
    def __enter__(self): 
        env = getattr(currentframe() , 'f_back')
        while env.f_code.co_name == '__enter__': env = getattr(env , 'f_back')
        assert env.f_code.co_name.startswith('on_') , env.f_code.co_name
        self.hook_name = env.f_code.co_name
        pass
    def __exit__(self): 
        pass