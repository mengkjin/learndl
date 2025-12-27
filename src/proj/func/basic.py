from src.proj.log import Logger
from pytimedinput import timedInput

__all__ = ['update_dict' , 'ask_for_confirmation']

def update_dict(old : dict , update : dict | None , recursive = True) -> dict:
    if update:
        if recursive:
            for k , v in update.items():
                if isinstance(v , dict) and isinstance(old.get(k) , dict):
                    old[k] = update_dict(old[k] , v)
                else:
                    old[k] = v
        else:
            old.update(update)
    return old

def ask_for_confirmation(prompt ='' , timeout = 10 , recurrent = 1 , proceed_condition = lambda x:True , print_function = Logger.stdout):
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
