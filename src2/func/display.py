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