"""Process discovery, power profile (non-Windows), and CLI arg parsing helpers."""
from __future__ import annotations
from typing import Any , Literal

from src.proj.log import Logger

__all__ = ['AskFor']

class AskFlag:
    """
    Ask for confirmation, selections, or retry.
    
    Args:
        flag : Literal['yes' , 'no' , 'abort']
        result : Any = None
        
    Returns:
        AskFlag object with the following properties:
        - yes : bool
        - no : bool
        - abort : bool
        - result : Any
          - list of selections for ask_for_selections
    """
    def __init__(
        self , 
        flag : Literal['yes' , 'no' , 'abort'] ,
        result : Any = None):
        self.flag : Literal['yes' , 'no' , 'abort'] = flag
        self.result : Any = result

    def __repr__(self) -> str:
        return f'AskFlag({self.flag})'
    def __str__(self) -> str:
        return self.flag

    @property
    def yes(self) -> bool:
        return self.flag == 'yes'
    @property
    def no(self) -> bool:
        return self.flag == 'no'
    @property
    def abort(self) -> bool:
        return self.flag == 'abort'

    def __bool__(self) -> bool:
        return self.yes

class AskFor:
    """
    Ask for confirmation, selections, or retry.
    example:
        flag = AskFor.Confirmation('Are you sure to continue?')
        flag = AskFor.Selections('Which model to archive?' , len(model_paths))
        flag = AskFor.Retry('Do you want to archive more models?')
    """
    @staticmethod
    def Confirmation(msg = '' , timeout = -1 , ask_times = 1):
        """Prompt up to ``recurrent`` times with optional per-prompt timeout.

        Returns:
            Tuple of (inputs list, bool list from ``proceed_condition``).
        """
        
        from pytimedinput import timedInput
        for i in range(ask_times):
            prefix = f'{msg} Please confirm (y/n/q) ({i+1}/{ask_times} rounds): '
                
            value, is_timeout = None , False
            if timeout > 0:
                try:
                    value, is_timeout = timedInput(f'{prefix} (in {timeout} seconds): ' , timeout = timeout)
                except Exception:
                    pass
            if value is None : 
                value, is_timeout = input(f'{prefix} : ') , False
            if value.lower() not in ['y' , 'n' , 'q']:
                Logger.error(f'Invalid input: {value}')
                return AskFlag('no')
            if is_timeout:
                Logger.stdout(f'Input is timed out at the {i+1}th round.')
                return AskFlag('no')
            elif value.lower() in ['n' , 'q']:
                Logger.stdout(f'Confirmation is rejected at the {i+1}th round.')
                return AskFlag('no')
            
        return AskFlag('yes')

    @staticmethod
    def Selections(msg : str , options : int , start : int = 1) -> AskFlag:
        """Parse the selections."""
        min , max = start , options + start - 1
        
        selection = input(f'{msg} ({min}-{max}, seperated by comma , q to quit): ')
        if selection.lower() == 'q':
            return AskFlag('no')
        selections = [s.strip() for s in selection.split(',') if s.strip()]
        if any(not s.isdigit() for s in selections):
            Logger.error(f'Contains non-digit characters: {selection}')
            return AskFlag('abort')

        selections = [int(i) for i in selections]
        if any(s < start or s > options + start - 1 for s in selections):
            Logger.error(f'Contains indices out of range [{min}-{max}]: {selection}')
            return AskFlag('abort')

        flag = input(f'Are you sure to select {selections}? (press y to confirm): ')
        if flag.lower() == 'y':
            return AskFlag('yes' , result = selections)
        else:
            return AskFlag('abort')

    @staticmethod
    def Retry(msg : str) -> AskFlag:
        """Ask for exit."""
        while True:
            flag = input(f'{msg} (y/n/q): ')
            if flag.lower() in ['n' , 'q']:
                return AskFlag('no')
            elif flag.lower() == 'y':
                return AskFlag('yes')
            else:
                Logger.error(f'Invalid input: {flag}')
