from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass , field
from datetime import datetime
from typing import Literal , Any

from src.proj import Logger
from src.res.model.util.core import epoch_key , attempt_key
from .pipeline import BasePipeline

EventTypeType = Literal[
    'new_attempt' , 'redo_attempt' , 'end_attempt' , 
    'recall_ckpt' , 'new_phase_recall' , 'new_phase' , 
    'milestone' , 'logging']
@dataclass
class EpochEvent:
    """Epoch event class, used to store the event of the epoch"""
    type : EventTypeType
    reason : str
    epoch : int
    phase : int = 0
    effective_epoch : int = -1
    message : str = ''
    details : dict[str,Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.effective_epoch < 0:
            self.effective_epoch = self.epoch

    def __repr__(self):
        return f'EpochEvent(type={self.type}, reason={self.reason}, epoch={self.epoch}, phase={self.phase}, effective_epoch={self.effective_epoch}, message={self.message}, details={self.details}, vb_level={self.vb_level})'

    @property
    def vb_level(self) -> Any:
        if self.type in ['new_attempt' , 'redo_attempt' , 'end_attempt' , 'recall_ckpt' , 'new_phase_recall' , 'new_phase']:
            return 1
        elif self.type in ['milestone']:
            return 2
        elif self.type in ['logging']:
            return 3
        else:
            raise ValueError(f'Invalid event type: {self.type}')

    @property
    def info(self) -> str:
        if self.message:
            return self.message
        elif 'info' in self.details:
            return self.details['info']
        else:
            info = f'Event {self.type} : [{self.reason}] at epoch {self.epoch}'
            if self.effective_epoch != self.epoch:
                info += f', effective at epoch {self.effective_epoch}'
            return info
            
@dataclass
class EpochRecord:
    """Epoch event class, used to store the event of the epoch"""
    attempt : int = 0
    phase : int = 0
    epoch : int = 0
    redo : int = 0
    events : list[EpochEvent] = field(default_factory=list)
    details : dict[str,Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.attempt >= 0 , 'attempt should be non-negative'
        assert self.phase >= 0 , 'phase should be non-negative'
        assert self.epoch >= -1 , 'epoch should be non-negative'
        assert self.redo >= 0 , 'redo should be non-negative'

    @property
    def epoch_key(self) -> str:
        return epoch_key(self.epoch , self.phase)

    @property
    def attempt_key(self) -> str:
        return attempt_key(self.attempt , self.redo)

    @property
    def continue_status(self) -> Literal['continue' , 'redo_attempt' , 'new_attempt' , 'end_attempt' , 'new_phase_recall' , 'new_phase' , 'recall_ckpt']:
        event_types = list(set([event.type for event in self.events]))
        if 'redo_attempt' in event_types:
            return 'redo_attempt'
        if 'new_attempt' in event_types:
            return 'new_attempt'
        if 'end_attempt' in event_types:
            return 'end_attempt'
        if 'new_phase_recall' in event_types:
            return 'new_phase_recall'
        if 'new_phase' in event_types:
            return 'new_phase'
        if 'recall_ckpt' in event_types:
            return 'recall_ckpt'
        return 'continue'

    def find_event(self , event_type : EventTypeType , allow_multiple : bool = False) -> EpochEvent | None:
        """find the first event of the given type"""
        events = [event for event in self.events if event.type == event_type]
        assert allow_multiple or len(events) <= 1 , f'Only one {event_type} event is allowed, but got {len(events)}'
        return events[0] if events else None

    def add_event(self , event : EpochEvent):
        self.events.append(event)

    def next_attempt(self , status : str | None = None) -> int:
        status = status or self.continue_status
        if status == 'new_attempt':
            return self.attempt + 1
        else:
            return self.attempt

    def next_redo(self , status : str | None = None) -> int:
        status = status or self.continue_status
        if status == 'redo_attempt':
            return self.redo + 1
        else:
            return self.redo

    def next_phase(self , status : str | None = None) -> int:
        status = status or self.continue_status
        if status in ['new_phase' , 'new_phase_recall']:
            return self.phase + 1
        elif status in ['new_attempt' , 'redo_attempt' , 'end_attempt']:
            return 0
        else:
            return self.phase

    def next_epoch(self , status : str | None = None) -> int:
        status = status or self.continue_status
        if status in ['continue' , 'new_phase']:
            return self.epoch + 1
        elif status in ['new_phase_recall' , 'recall_ckpt']:
            recall_events = [event for event in self.events if event.type == status]
            assert len(recall_events) == 1 , f'Only one recall_ckpt event is allowed, but got {len(recall_events)}'
            return recall_events[0].effective_epoch + 1
        elif status in ['new_attempt' , 'redo_attempt' , 'end_attempt']:
            return 0
        else:
            raise ValueError(f'Invalid continue status: {status}')

    def new_epoch(self) -> EpochRecord | None:
        status = self.continue_status
        attempt = self.next_attempt(status)
        redo = self.next_redo(status)
        phase = self.next_phase(status)
        epoch = self.next_epoch(status)
        if status == 'end_attempt':
            return None
        else:
            return EpochRecord(attempt , phase , epoch , redo)

class FittingEpochs:
    """Fitting epochs class, used to store the epochs of the fitting"""
    def __init__(self , max_epoch : int = 200):
        self.max_epoch = max_epoch
        self.epochs : list[EpochRecord] = [EpochRecord(epoch = -1)]
        self.events : dict[EventTypeType , list[EpochEvent]] = defaultdict(list)
        self.attempt_events : dict[EventTypeType , list[EpochEvent]] = defaultdict(list)
    def __len__(self):
        return len(self.epochs)
    def __repr__(self):
        return f'FittingEpochs(epochs={len(self.epochs)}, events=({", ".join([f"{k}={len(v)}" for k,v in self.events.items()])}))'
    def __bool__(self):
        return bool(self.epochs)
    def clear(self):
        """clear the epochs and events every time a new model is started, but keep the attempt events"""
        self.epochs.clear()
        self.epochs.append(EpochRecord(epoch = -1))
        self.events.clear()
        self.attempt_events.clear()
    def __getitem__(self , index : int) -> EpochRecord:
        return self.epochs[index]
    @property
    def current(self) -> EpochRecord:
        return self.epochs[-1]

    @property
    def next_attempt(self) -> int:
        return self.current.next_attempt()
    @property
    def next_redo(self) -> int:
        return self.current.next_redo()

    @property
    def model_epoch(self) -> int:
        return len(self.epochs) - 1
    @property
    def loop_end(self) -> bool:
        return self.current.continue_status == 'end_attempt'
    def new_epoch(self):
        if self.current.continue_status in ['redo_attempt' , 'new_attempt']:
            for type in self.attempt_events:
                self.events[type].extend(self.attempt_events[type])
            self.attempt_events.clear()
        record = self.current.new_epoch() if self.epochs else EpochRecord()
        assert record is not None , 'record is None'
        self.epochs.append(record)
    def add_epoch_event(self , type : EventTypeType , reason : str , epoch : int | None = None , message : str = '' , details : dict[str,Any] | None = None):
        effective_epoch = -1 if epoch is None else epoch
        current_epoch = self.current
        event = EpochEvent(type , reason , current_epoch.epoch , current_epoch.phase , effective_epoch , message , details or {})
        self.attempt_events[type].append(event)
        current_epoch.add_event(event)
    def check_loop_end(self):
        if self.current.epoch >= self.max_epoch - 1:
            self.add_epoch_event('end_attempt' , 'Max Epoch' , message = f'Max Epoch ({self.max_epoch}) reached, force end attempt')
    @property
    def end_attempt_event(self) -> EpochEvent | None:
        if not self.attempt_events['end_attempt']:
            return None
        event = sorted(self.attempt_events['end_attempt'] , key = lambda x: (x.effective_epoch , x.epoch))[0]
        return event
    
class TrainerStatus(BasePipeline):
    """Trainer status class, used to store the status of the trainer"""
    def __init__(self , max_epoch : int):
        self.stage   : Literal['data' , 'fit' , 'test'] = 'data'
        self.dataset : Literal['train' , 'valid' , 'test' , 'predict'] = 'train'
        self.model_num  : int = -1
        self.model_date : int = -1
        self.model_submodel : str = 'best'

        self.fitting_epochs : FittingEpochs = FittingEpochs(max_epoch)
        self.milestone_epochs : list[int] = []
        self.total_epochs : int = 0
        self.total_models : int = 0
        self.times : dict[str,datetime] = {}
        
    def __repr__(self):
        return f'TrainerStatus({", ".join([f"{k}={v}" for k,v in self.status.items()])})'

    def __getitem__(self , i : int) -> EpochRecord:
        return self.fitting_epochs[i]

    @property
    def status(self):
        return {
            'stage' : self.stage ,
            'dataset' : self.dataset ,
            'model_num' : self.model_num ,
            'model_date' : self.model_date ,
            'model_submodel' : self.model_submodel ,
            'attempt' : self.attempt ,
            'redo' : self.redo ,
            'phase' : self.phase ,
            'epoch' : self.epoch ,
            'next_attempt' : self.fitting_epochs.next_attempt ,
            'next_redo' : self.fitting_epochs.next_redo ,
            'epoch_key' : self.epoch_key ,
            'model_epoch' : self.model_epoch ,
        }
    @property
    def loop_end(self) -> bool:
        return self.fitting_epochs.loop_end
    @property
    def end_attempt_event(self) -> EpochEvent | None:
        return self.fitting_epochs.end_attempt_event
    @property
    def current(self) -> EpochRecord:
        return self.fitting_epochs.current
    @property
    def attempt(self) -> int:
        return self.current.attempt
    @property
    def redo(self) -> int:
        return self.current.redo
    @property
    def phase(self) -> int:
        return self.current.phase
    @property
    def epoch(self) -> int:
        return self.current.epoch
    @property
    def epoch_key(self) -> str:
        return self.current.epoch_key
    @property
    def attempt_key(self) -> str:
        return self.current.attempt_key
    @property
    def next_attempt(self) -> int:
        return self.current.next_attempt()
    @property
    def next_redo(self) -> int:
        return self.current.next_redo()
    @property
    def next_attempt_key(self) -> str:
        return attempt_key(self.next_attempt , self.next_redo)
    @property
    def model_epoch(self) -> int:
        return self.fitting_epochs.model_epoch
    @property
    def milestone_epoch(self) -> int:
        return 0 if not self.milestone_epochs else self.milestone_epochs[-1]
    @property
    def model_status(self) -> str:
        return f'{self.model_num}.{self.model_date}.{self.attempt_key}'

    def set_milestone_epoch(self , epoch : int):
        self.milestone_epochs.append(epoch)
    def add_epoch_event(self , type : EventTypeType , reason : str , epoch : int | None = None , message : str = '' , details : dict[str,Any] | None = None):
        self.fitting_epochs.add_epoch_event(type , reason , epoch , message , details)
    
    def on_data_start_before(self):  
        self.stage = 'data'
        self.times['data_start'] = datetime.now()
    def on_data_end_after(self):    
        self.times['data_end'] = datetime.now()
    def on_fit_start_before(self):    
        self.stage = 'fit'
        self.times['fit_start'] = datetime.now()
    def on_fit_end_after(self):        
        self.times['fit_end'] = datetime.now()
    def on_test_start_before(self):    
        self.stage = 'test'
        self.times['test_start'] = datetime.now()
    def on_test_end_after(self):      
        self.times['test_end'] = datetime.now()
    def on_train_epoch_start(self): 
        self.dataset = 'train'
    def on_validation_epoch_start(self): 
        self.dataset = 'valid'
    def on_test_model_start(self): 
        self.dataset = 'test'
    def on_fit_model_start(self):
        Logger.only_once(f'In Stage [{self.stage}], First Iterance: ({self.model_date} , {self.model_num})' , object = self , printer = Logger.note)
        self.times['model_start'] = datetime.now()
        self.fitting_epochs.clear()
        self.milestone_epochs.clear()
        self.total_models += 1
    def on_fit_epoch_start(self):
        self.fitting_epochs.new_epoch()
        self.total_epochs += 1
    def on_fit_epoch_end_before(self):
        self.fitting_epochs.check_loop_end()    