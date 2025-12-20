import pandas as pd
from matplotlib.figure import Figure
from IPython.display import display as raw_display
from typing import Callable

__all__ = ['Display']

class Display:
    """Display the object in the best way"""
    _instance = None
    _callbacks_before : list[Callable] = []
    _callbacks_after : list[Callable] = []
    def __new__(cls , *args, **kwargs):
        """only one instance of the display will be created"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , obj = None , raise_error = True):
        """
        display the object if it is not None in the constructor
        example:
            Display(obj) # will create (or use the existing instance) a new instance of the display and also display the object
        """
        if obj is not None:
            self(obj , raise_error = raise_error)

    def __call__(self , obj , raise_error = True):
        """
        display the object
        """
        assert obj is not None , "No object to display"
        self.display(obj , raise_error = raise_error)

    @classmethod
    def set_callbacks(cls , callbacks_before : list[Callable] | None = None, callbacks_after : list[Callable] | None = None):
        """
        set the callbacks before and after the display
        example:
            Display.set_callbacks(callback_before , callback_after)
            means:
                before the display, the callback_before will be called
                after the display, the callback_after will be called
        """
        if callbacks_before is not None:
            cls._callbacks_before.extend(callbacks_before)
        if callbacks_after is not None:
            cls._callbacks_after.extend(callbacks_after)

    @classmethod
    def reset_callbacks(cls):
        """
        reset the callbacks before and after the display
        """
        cls._callbacks_before.clear()
        cls._callbacks_after.clear()

    @classmethod
    def display(cls , obj , raise_error = True):
        """
        display the object
        """
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
        """
        display a pandas dataframe
        """
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
        """
        display a matplotlib figure
        """
        if fig is None:
            if raise_error: 
                raise ValueError('No figure to display')
            return
        raw_display(fig)