import logging

from inspect import currentframe

from ..config import TrainConfig
from ..metric import Metrics 
from ...classes import BaseCB , TrainerStatus 

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
        pass
    def __exit__(self): 
        pass