from typing import Callable , TypeVar
from src.proj import Logger

T = TypeVar('T')

def wrap_update(update_func : Callable[..., T] , message : str , skip : bool = False , *args , **kwargs) -> T | None:
    '''
    Internal helper: run *update_func* inside a Logger paragraph; skip logs a warning instead.

    Not intended as a user-facing Streamlit API endpoint (used only by other ``src.api`` callables).

    Args:
        update_func: Callable to run when ``skip`` is false.
        message: Log section title.
        skip: If true, skip execution and log a skip warning.
    '''
    if skip:
        Logger.warning(f'Process [{message.title()}] is Skipped')
    else:
        with Logger.Paragraph(message , 3):
            return update_func(*args , **kwargs)
        