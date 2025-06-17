import pandas as pd

from IPython.display import display 

def data_frame(df : pd.DataFrame , text_ahead : str | None = None , text_after : str | None = None):
    if text_ahead: print(text_ahead)
    with pd.option_context(
        'display.max_rows', 100,
        'display.max_columns', None,
        'display.width', 1000,
        'display.precision', 3,
        'display.colheader_justify', 'center'):
        display(df)
    if text_after: print(text_after)

def plot(fig , raise_error = True):
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

