import pandas as pd
from matplotlib.figure import Figure
from IPython.display import display as raw_display
from typing import Callable

__all__ = ['Display']

class Display:
    _instance = None
    _callbacks_before : list[Callable] = []
    _callbacks_after : list[Callable] = []
    def __new__(cls , *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , obj = None , raise_error = True):
        if obj is not None:
            self(obj , raise_error = raise_error)

    def __call__(self , obj , raise_error = True):
        assert obj is not None , "No object to display"
        self.display(obj , raise_error = raise_error)

    @classmethod
    def set_callbacks(cls , callbacks_before : list[Callable] | None = None, callbacks_after : list[Callable] | None = None):
        if callbacks_before is not None:
            cls._callbacks_before.extend(callbacks_before)
        if callbacks_after is not None:
            cls._callbacks_after.extend(callbacks_after)

    @classmethod
    def reset_callbacks(cls):
        cls._callbacks_before.clear()
        cls._callbacks_after.clear()

    @classmethod
    def display(cls , obj , raise_error = True):
        for callback in cls._callbacks_before:
            callback(obj)
            
        if isinstance(obj , Figure):
            cls.figure(obj , raise_error = raise_error)
        elif isinstance(obj , pd.DataFrame):
            cls.data_frame(obj , raise_error = raise_error)
        else:
            raw_display(obj)

        for callback in cls._callbacks_after:
            callback(obj)

    @staticmethod
    def data_frame(df : pd.DataFrame , raise_error = True):
        if df is None:
            if raise_error: 
                raise ValueError('No dataframe to display')
            return
        with pd.option_context(
            'display.max_rows', 100,
            'display.max_columns', None,
            'display.width', 1000,
            'display.precision', 3,
            'display.colheader_justify', 'center'):
            raw_display(df)

    @staticmethod
    def figure(fig : Figure , raise_error = True):
        if fig is None:
            if raise_error: 
                raise ValueError('No figure to display')
            return
        raw_display(fig)