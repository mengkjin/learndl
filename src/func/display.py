import pandas as pd
from matplotlib.figure import Figure
from IPython.display import display as raw_display

__all__ = ['display' , 'EnclosedMessage' , 'print_seperator']

def display(obj , raise_error = True):
    if isinstance(obj , Figure):
        display_figure(obj)
    elif isinstance(obj , pd.DataFrame):
        display_data_frame(obj , raise_error = raise_error)
    else:
        raw_display(obj)

def display_data_frame(df , raise_error = True):
    if df is None:
        if raise_error: raise ValueError('No dataframe to display')
        return
    with pd.option_context(
        'display.max_rows', 100,
        'display.max_columns', None,
        'display.width', 1000,
        'display.precision', 3,
        'display.colheader_justify', 'center'):
        display(df)

def display_figure(fig , raise_error = True):
    if fig is None:
        if raise_error: raise ValueError('No figure to display')
        return
    display(fig)

def print_seperator(width = 80 , char = '-'):
    print(char * width)

class EnclosedMessage:
    def __init__(self , title : str , width = 80):
        self.title = title
        self.width = width

    def __enter__(self):
        print_seperator(self.width)
        if len(self.title) >= self.width:
            print(self.title.upper())
        else:
            padding = '*' * ((self.width - len(self.title)) // 2)
            print(padding + self.title.upper() + padding)

    def __exit__(self , exc_type , exc_value , traceback):
        print_seperator(self.width)

