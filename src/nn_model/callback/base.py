import logging

from ..util import Checkpoint , Metrics , TrainConfig
from ..basic import BaseCB , TrainerStatus , BaseDataModule , BaseTrainer

class CallBack(BaseCB):
    def __init__(self , model_module : BaseTrainer , with_cb : bool , print_info = True , *args , **kwargs) -> None:
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
    @property
    def batch_data(self): return self.module.batch_data
    @property
    def batch_idx(self): return self.module.batch_idx
    @property
    def batch_output(self): return self.module.batch_output
