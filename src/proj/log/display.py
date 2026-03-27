import pandas as pd
from matplotlib.figure import Figure
from IPython.display import display as raw_display
from typing import Any , Callable

from src.proj.proj import Proj

__all__ = ['Display']

class Display:
    """Display the object in the best way , vb_level can be set to control display"""
    _instance = None
    _callbacks_before : list[Callable] = []
    _callbacks_after : list[Callable] = []
    def __new__(cls , *args, **kwargs):
        """only one instance of the display will be created"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self , obj = None , vb_level : Any = 1 , **kwargs):
        """
        display the object if it is not None in the constructor
        example:
            Display(obj) # will create (or use the existing instance) a new instance of the display and also display the object
        """
        if obj is not None:
            self(obj , vb_level = vb_level , **kwargs)

    def __call__(self , obj , **kwargs):
        """
        display the object
        """
        assert obj is not None , "No object to display"
        self.display(obj , **kwargs)

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
    def display(cls , obj , vb_level : Any = 1 , **kwargs):
        """
        display the object
        """
        if Proj.silence.silent or Proj.vb.ignore(vb_level):
            return
        with Proj.vb.WithVbLevel(vb_level):
            for callback in cls._callbacks_before:
                callback(obj)
                
            if isinstance(obj , Figure):
                cls.figure(obj , **kwargs)
            elif isinstance(obj , pd.DataFrame):
                cls.data_frame(obj , **kwargs)
            else:
                raw_display(obj)

            for callback in cls._callbacks_after:
                callback(obj)

    @staticmethod
    def data_frame(df : pd.DataFrame , **kwargs):
        """
        display a pandas dataframe
        """
        with pd.option_context(
            'display.max_rows', 100,
            'display.max_columns', None,
            'display.width', 1000,
            'display.precision', 3,
            'display.colheader_justify', 'center' ,
            *[i for k,v in kwargs.items() for i in [k,v]]):
            raw_display(df)

    @staticmethod
    def figure(fig : Figure , **kwargs):
        """
        display a matplotlib figure
        """
        raw_display(fig)