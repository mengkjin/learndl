"""
Utility functions for api operations of this project, mostly used for update functions.
"""

from __future__ import annotations
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import TypeVar
from collections.abc import Callable
from src.proj import Logger , Base

T = TypeVar('T')

__all__ = ['wrap_update' , 'print_update_records']

_UpdateRecords : list[_UpdateWrapperRecord] = []
@dataclass
class _UpdateWrapperRecord:
    """Internal wrapper record for update functions."""
    update_func : Callable
    message : str
    start_time : datetime
    end_time : datetime | None = None
    skip : bool = False

    def close(self):
        """Close the record."""
        self.end_time = datetime.now()

    @classmethod
    def start(cls , update_func : Callable , message : str , skip : bool = False):
        """Start a record."""
        record = cls(update_func , message , datetime.now() , None , skip)
        _UpdateRecords.append(record)
        return record

    def duration_str(self) -> str:
        """Get the duration string."""
        assert self.end_time is not None , 'end_time is not set'
        return Base.Elapsed(self.end_time - self.start_time).fmtstr

    def to_dict(self) -> dict:
        """Convert the record to a dictionary."""
        return {
            'name': self.update_func.__name__,
            'message': self.message,
            'start': self.start_time,
            'end': self.end_time,
            'duration': self.duration_str(),
        }

def wrap_update(update_func : Callable[..., T] , message : str , skip : bool = False , *args , **kwargs) -> T | None:
    """
    Internal helper: run *update_func* inside a Logger paragraph; skip logs a warning instead.

    Not intended as a user-facing Streamlit API endpoint (used only by other ``src.api`` callables).

    Args:
        update_func: Callable to run when ``skip`` is false.
        message: Log section title.
        skip: If true, skip execution and log a skip warning.
    """
    record = _UpdateWrapperRecord.start(update_func , message , skip)
    if skip:
        Logger.warning(f'Process [{message.title()}] is Skipped')
    else:
        with Logger.Paragraph(message , 3):
            ret = update_func(*args , **kwargs)
    record.close()
    return ret

def print_update_records():
    """Print the update records during this process."""
    record_df = pd.DataFrame([record.to_dict() for record in _UpdateRecords if record.end_time is not None and not record.skip])
    Logger.display(record_df , title = 'Update Records')