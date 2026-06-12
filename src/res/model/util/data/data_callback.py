"""
Data callbacks for the data module
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from src.res.model.util.core import BatchInput

__all__ = ['DataCallbacks']

class DataCallbacks:
    """callbacks for data module"""
    def __init__(self):
        self.callbacks = defaultdict(list)
    def register_callbacks(self , hook_name : str , *callbacks : Callable):
        assert hook_name in ['on_before_batch_transfer' , 'on_after_batch_transfer'] , hook_name
        for callback in callbacks:
            self.callbacks[hook_name].append(callback)
    def on_before_batch_transfer(self , batch : BatchInput) -> BatchInput: 
        for callback in self.callbacks['on_before_batch_transfer']:
            batch = callback(batch)
        return batch
    def on_after_batch_transfer(self , batch : BatchInput) -> BatchInput: 
        for callback in self.callbacks['on_after_batch_transfer']:
            batch = callback(batch)
        return batch