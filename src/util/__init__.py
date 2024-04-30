from . import (
    callback , config , device , ensemble , loader , logger , metric , optim , store , time
)

from .callback import CallBackManager
from .config import TrainConfig
from .device import Device
from .ensemble import FittestModel
from .loader import DataloaderStored , LoaderWrapper
from .logger import Logger
from .metric import Metrics , MetricsAggregator
from .optim import Optimizer
from .store import Checkpoint , Deposition , Storage
from .time import PTimer

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