"""Small utilities: dict merge, timed confirmation prompts, name casing."""

from src.proj.log import Logger
from pytimedinput import timedInput

__all__ = ['ask_for_confirmation']

def ask_for_confirmation(prompt ='' , timeout = 10 , recurrent = 1 , proceed_condition = lambda x:True , print_function = Logger.stdout):
    """Prompt up to ``recurrent`` times with optional per-prompt timeout.

    Returns:
        Tuple of (inputs list, bool list from ``proceed_condition``).
    """
    assert isinstance(prompt , str) , prompt
    userText_list , userText_cond = [] , []
    for t in range(recurrent):
        if t == 0:
            _prompt = prompt 
        elif t == 1:
            _prompt = 'Really?'
        else:
            _prompt = 'Really again?'
            
        userText, timedOut = None , None
        if timeout > 0:
            try:
                userText, timedOut = timedInput(f'{_prompt} (in {timeout} seconds): ' , timeout = timeout)
            except Exception:
                pass
        if userText is None : 
            userText, timedOut = input(f'{_prompt} : ') , False
        (_timeout , _sofar) = ('Time Out! ' , 'so far') if timedOut else ('' , '')
        print_function(f'{_timeout}User-input {_sofar} is : [{userText}].')
        userText_list.append(userText)
        userText_cond.append(proceed_condition(userText))
        if not userText_cond[-1]: 
            break
    return userText_list , userText_cond
