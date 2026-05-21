from __future__ import annotations
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Callable , TypeVar
from src.proj import Logger
from src.proj.core import Duration

T = TypeVar('T')

_UpdateRecords : list[UpdateWrapperRecord] = []
@dataclass
class UpdateWrapperRecord:
    update_func : Callable
    message : str
    start_time : datetime
    end_time : datetime | None = None
    skip : bool = False

    def close(self):
        self.end_time = datetime.now()

    @classmethod
    def start(cls , update_func : Callable , message : str , skip : bool = False):
        record = UpdateWrapperRecord(update_func , message , datetime.now() , None , skip)
        _UpdateRecords.append(record)
        return record

    def duration_str(self) -> str:
        assert self.end_time is not None , 'end_time is not set'
        return Duration(self.end_time - self.start_time).fmtstr

    def to_dict(self) -> dict:
        return {
            'name': self.update_func.__name__,
            'message': self.message,
            'start': self.start_time,
            'end': self.end_time,
            'duration': self.duration_str(),
        }

def wrap_update(update_func : Callable[..., T] , message : str , skip : bool = False , *args , **kwargs) -> T | None:
    '''
    Internal helper: run *update_func* inside a Logger paragraph; skip logs a warning instead.

    Not intended as a user-facing Streamlit API endpoint (used only by other ``src.api`` callables).

    Args:
        update_func: Callable to run when ``skip`` is false.
        message: Log section title.
        skip: If true, skip execution and log a skip warning.
    '''
    record = UpdateWrapperRecord.start(update_func , message , skip)
    if skip:
        Logger.warning(f'Process [{message.title()}] is Skipped')
    else:
        with Logger.Paragraph(message , 3):
            return update_func(*args , **kwargs)
    record.close()
        

def print_update_records():
    record_df = pd.DataFrame([record.to_dict() for record in _UpdateRecords if record.end_time is not None and not record.skip])
    Logger.display(record_df , caption = 'Update Records')