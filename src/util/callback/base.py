import logging
from typing import Callable
from ..config import TrainConfig
from ..metric import Metrics 
from ..store import Checkpoint
from ...classes import BaseCB , TrainerStatus , BaseDataModule , BaseTrainerModule

class CallBack(BaseCB):
    def __init__(self , model_module : BaseTrainerModule , with_cb : bool , print_info = True , *args , **kwargs) -> None:
        super().__init__(model_module , with_cb)
        if print_info: self._print_info(depth=1)
    @property
    def config(self) -> TrainConfig:    return self.module.config
    @property
    def logger(self) -> logging.Logger: return self.module.logger
    @property
    def status(self) -> TrainerStatus:  return self.module.status
    @property
    def metrics(self) -> Metrics :  return self.module.metrics
    @property
    def ckpt(self) -> Checkpoint: return self.module.checkpoint
    @property
    def data(self) -> BaseDataModule: return self.module.data