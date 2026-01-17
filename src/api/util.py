from src.proj import Logger

def wrap_update(update_func , message : str , skip : bool = False , *args , **kwargs):
    '''
    Wrap update function with logger enclosed message and warning if skip
    '''
    if skip:
        Logger.warning(f'Process [{message.title()}] is Skipped')
    else:
        with Logger.Paragraph(message , 3):
            update_func(*args , **kwargs)
        