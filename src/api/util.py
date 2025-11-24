from src.proj import Logger

def wrap_update(update_func , message : str , skip : bool = False , *args , **kwargs):
    '''
    Wrap update function with logger enclosed message and warning if skip
    '''
    if skip:
        Logger.warning(f'Skipping: {message.title()}')
    else:
        with Logger.EnclosedMessage(f' {message.title()} '):
            update_func(*args , **kwargs)
        