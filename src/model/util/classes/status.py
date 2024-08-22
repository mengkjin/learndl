import numpy as np

from dataclasses import dataclass , field
from typing import Any , Literal , Optional


@dataclass
class TrainerStatus:
    max_epoch : int = 200
    stage   : Literal['data' , 'fit' , 'test'] = 'data'
    dataset : Literal['train' , 'valid' , 'test'] = 'train'
    epoch   : int = -1
    attempt : int = 0
    round   : int = 0
    
    model_num  : int = -1
    model_date : int = -1
    model_type : str = 'best'
    epoch_event : list[str] = field(default_factory=list)
    best_attempt_metric : Any = None

    def __post_init__(self):
        self.end_of_loop = self.EndofLoop(self.max_epoch)
        self.fit_iter_num :int = 0

    def stage_data(self): self.stage = 'data'
    def stage_fit(self):  self.stage = 'fit'
    def stage_test(self): self.stage = 'test'
    def dataset_train(self): self.dataset = 'train'
    def dataset_validation(self): self.dataset = 'valid'
    def dataset_test(self): self.dataset = 'test'
    def fit_model_start(self):
        self.fit_iter_num += 1
        self.attempt = -1
        self.best_attempt_metric = None
        self.new_attempt()
    def fit_epoch_start(self):
        self.epoch   += 1
        self.epoch_event = []
    def fit_epoch_end(self):
        self.end_of_loop.loop_end(self.epoch)
    def new_attempt(self , event : Literal['new_attempt' , 'nanloss'] = 'new_attempt'):
        self.epoch   = -1
        self.round   = 0
        self.end_of_loop = self.EndofLoop(self.max_epoch)
        self.epoch_event = []

        self.add_event(event)
        if event == 'new_attempt': self.attempt += 1

    def add_event(self , event : Optional[str]):
        if event: self.epoch_event.append(event)

    @dataclass
    class EndofLoop:
        max_epoch : int = 200
        status : list['EndStatus'] = field(default_factory=list)

        @dataclass
        class EndStatus:
            name  : str
            epoch : int # epoch of trigger

        def __post_init__(self) -> None: ...
        def __bool__(self): return len(self.status) > 0
        def new_loop(self): self.status = []
        def loop_end(self , epoch):
            if epoch >= self.max_epoch: self.add_status('Max Epoch' , epoch)
        def add_status(self , status : str , epoch : int): 
            self.status.append(self.EndStatus(status , epoch))
        @property
        def end_epochs(self) -> list[int]:
            return [sta.epoch for sta in self.status]
        @property
        def trigger_i(self) -> int:
            return np.argmin(self.end_epochs).item()
        @property
        def trigger_ep(self) -> int:
            return self.status[self.trigger_i].epoch
        @property
        def trigger_reason(self):
            return self.status[self.trigger_i].name