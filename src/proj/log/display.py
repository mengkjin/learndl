"""IPython-friendly display routing (DataFrame, matplotlib Figure, or fallback)."""

from typing import Callable

__all__ = ['Display']

class Display:
    """Display the object in the best way"""
    _callbacks_before : list[Callable] = []
    _callbacks_after : list[Callable] = []

    def __init__(self , obj = None , **kwargs):
        """
        display the object if it is not None in the constructor
        example:
            Display(obj) # will create (or use the existing instance) a new instance of the display and also display the object
        """
        if obj is not None:
            self(obj , **kwargs)

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
    def display(cls , obj , **kwargs):
        """
        display the object
        """
        for callback in cls._callbacks_before:
            callback(obj)
        from matplotlib.figure import Figure
        import pandas as pd
        if isinstance(obj , Figure):
            cls.figure(obj , **kwargs)
        elif isinstance(obj , pd.DataFrame):
            cls.data_frame(obj , **kwargs)
        else:
            cls.raw_display(obj)

        for callback in cls._callbacks_after:
            callback(obj)

    @classmethod
    def data_frame(cls , df , **kwargs):
        """
        display a pandas dataframe
        """
        import pandas as pd
        with pd.option_context(
            'display.max_rows', 100,
            'display.max_columns', None,
            'display.width', 1000,
            'display.precision', 3,
            'display.colheader_justify', 'center' ,
            *[i for k,v in kwargs.items() for i in [k,v]]):
            cls.raw_display(df)

    @classmethod
    def figure(cls , fig , **kwargs):
        """
        display a matplotlib figure
        """
        cls.raw_display(fig)

    @staticmethod
    def raw_display(obj):
        """
        display the object
        """
        from IPython.display import display as raw_display
        raw_display(obj)