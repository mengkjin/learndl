from .trainer import optim
from . import (
    config, device , loader , logger , metric, store , time , trainer
)

from .trainer.pipeline import Pipeline
from .trainer.ckpt import Checkpoint
from .trainer.model import FittestModel
from .trainer.optim import Optimizer
from .loader import DataloaderStored
from .metric import Metrics , AggMetrics
from .store import Storage
from .device import Device
from .logger import Logger
from .config import TrainConfig

class Filtered:
    def __init__(self, iterable, condition):
        self.iterable  = iter(iterable)
        self.condition = condition if callable(condition) else iter(condition)
    def __iter__(self):
        return self
    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: return item