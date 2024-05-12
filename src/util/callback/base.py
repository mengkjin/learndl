import logging
from typing import Callable
from ..config import TrainConfig
from ..metric import Metrics 
from ...classes import BaseCB , TrainerStatus , BaseDataModule , BaseModelModule

class BasicCallBack(BaseCB):
    def __init__(self , model_module : BaseModelModule , *args , **kwargs) -> None:
        super().__init__(model_module)
    @property
    def config(self) -> TrainConfig:    return self.module.config
    @property
    def logger(self) -> logging.Logger: return self.module.logger
    @property
    def status(self) -> TrainerStatus:  return self.module.status
    @property
    def metrics(self) -> Metrics :  return self.module.metrics
    @property
    def data_mod(self) -> BaseDataModule: return self.module.data_mod
            
class WithCallBack(BasicCallBack):
    def __init__(self, model_module, *args, **kwargs) -> None:
        super().__init__(model_module, *args, **kwargs)
        self.__hook_stack = []
    def hook_wrapper(self , hook : Callable):
        self.at_enter(hook.__name__)
        hook()
        self.at_exit(hook.__name__)
    def at_enter(self , hook_name): ...
    def at_exit(self , hook_name): self.__getattribute__(hook_name)()
    def __enter__(self): 
        self.__hook_stack.append(self.trace_hook_name)
        self.at_enter(self.__hook_stack[-1])
    def __exit__(self , *args): self.at_exit(self.__hook_stack.pop())

