"""Process discovery, power profile (non-Windows), and CLI arg parsing helpers."""
from __future__ import annotations

import sys
from typing import Any , Literal , Generator

from src.proj.log import Logger

__all__ = ['AskFor']

def _print_title(title : str):
    if title:
        Logger.stdout(f'{title}' , color = 'lightpurple')

AskFlagType = Literal['yes' , 'no' , 'abort']
class AskFlag:
    """
    Ask for confirmation, selections, or retry.
    
    Args:
        flag : AskFlagType
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
        flag : AskFlagType ,
        result : Any = None
    ):
        self.flag : AskFlagType = flag
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

class LoopFlag:
    """
    Loop flag for the loop until exit.
    """
    def __init__(self , round : int = 0 , flag : AskFlagType = 'yes'):
        self.round : int = round
        self._flag : AskFlag = AskFlag(flag = flag)

    def __repr__(self) -> str:
        return f'LoopFlag({self.round},{self._flag})'
    def __str__(self) -> str:
        return f'Round {self.round} , Flag: {self._flag}'

    def __bool__(self) -> bool:
        return self._flag.yes

    def set_flag(self , flag : AskFlag) -> LoopFlag:
        self._flag = flag
        return self

    @property
    def flag(self) -> AskFlag:
        return self._flag
class AskFor:
    """
    Ask for confirmation, selections, or retry.
    example:
        flag = AskFor.Confirmation('Are you sure to continue?')
        flag = AskFor.Selections('Which model to archive?' , len(model_paths))
        flag = AskFor.Options(['fit' , 'predict' , 'both'] , confirm = False , multiple = False , title = f'Which type of data to reconstruct? (fit/predict/both)')
        flag = AskFor.Retry('Do you want to archive more models?')
    """
   
    not_interactive = not sys.stdin.isatty()

    @classmethod
    def Confirmation(cls , timeout = -1 , ask_times = 1 , title = '') -> AskFlag:
        """Prompt up to ``recurrent`` times with optional per-prompt timeout.

        Returns:
            Tuple of (inputs list, bool list from ``proceed_condition``).
        """
        if cls.not_interactive:
            Logger.error('Not interactive mode, return false!')
            return AskFlag('no')
        assert ask_times > 0 , 'ask_times must be greater than 0'
        
        _print_title(title)
        for i in range(ask_times):
            prefix = f'Please press y to confirm' 
            if ask_times > 1:
                prefix += f' ({i+1}/{ask_times} rounds): '
            value, is_timeout = None , False
            if timeout > 0:
                try:
                    from pytimedinput import timedInput
                    value, is_timeout = timedInput(f'{prefix} (in {timeout} seconds): ' , timeout = timeout)
                    value = value.strip()
                except Exception:
                    pass
            if value is None : 
                value = input(f'{prefix} : ')
                is_timeout = False
            if is_timeout:
                Logger.stdout(f'Input is timed out at the {i+1}th round.')
                return AskFlag('no')
            if value.strip().lower() != 'y':
                Logger.error(f'Invalid input: {value} , confirmation is rejected at the {i+1}th round')
                return AskFlag('no') 
        return AskFlag('yes')

    @classmethod
    def Selections(
        cls , options : int | list[Any] , start : int = 1 , confirm : bool = True , 
        multiple : bool = False , title : str = ''
    ) -> AskFlag:
        """Ask for selections out of a number of options starting from a given index."""
        if cls.not_interactive:
            Logger.error('Not interactive mode, return false!')
            return AskFlag('no')
        if isinstance(options , int):
            num = options
            option_strs = [f'#{i+start:02d}' for i in range(num)]
        else:
            num = len(options)
            option_strs = [f'{i+start}.{option}' for i, option in enumerate(options)]
        min , max = start , num + start - 1
        _print_title(title)
        if multiple:
            selection = input(f'Choose from {min} to {max}, seperated by comma , q to quit: ')
            if selection.lower() == 'q':
                return AskFlag('no')
            choices = [s.strip() for s in selection.split(',') if s.strip()]
            if any(not s.isdigit() for s in choices):
                Logger.error(f'Contains non-digit characters: {selection}')
                return AskFlag('abort')

            choices = [int(i) for i in choices]
            if any(s < start or s > num + start - 1 for s in choices):
                Logger.error(f'Contains indices out of range [{min}-{max}]: {selection}')
                return AskFlag('abort')
            if len(choices) == 1:
                choices = choices[0]
        else:
            selection = input(f'Choose from {min} to {max}, q to quit: ')
            if selection.lower() == 'q':
                return AskFlag('no')
            if not selection.isdigit():
                Logger.error(f'Invalid input: {selection} , please choose from {min} to {max} or q to quit')
                return AskFlag('abort')
            choices = int(selection)
            if choices < start or choices > num + start - 1:
                Logger.error(f'Contains indices out of range [{min}-{max}]: {selection}')
                return AskFlag('abort')

        if confirm:
            flag = cls.Confirmation(title = f'Are you sure to select {option_strs[choices - start] if isinstance(choices , int) else choices}?')
            return AskFlag('yes' , result = choices) if flag.yes else AskFlag('abort')
        else:
            return AskFlag('yes' , result = choices)

    @classmethod
    def Retry(cls , title : str = '') -> AskFlag:
        """Ask for exit."""
        if cls.not_interactive:
            Logger.error('Not interactive mode, return false!')
            return AskFlag('no')
        _print_title(title)
        while True:
            value = input(f'Choose yes or no or quit (y/n/q): ')
            if value.strip().lower() in ['n' , 'q']:
                return AskFlag('no')
            elif value.strip().lower() == 'y':
                return AskFlag('yes')
            else:
                Logger.error(f'Invalid input: {value} , please choose yes or no or quit (y/n/q)')

    @classmethod
    def Options(
        cls , options : list[Any] , confirm : bool = True , 
        multiple : bool = False , title : str = ''
    ) -> AskFlag:
        """Ask for options from a list of options."""
        if cls.not_interactive:
            Logger.error('Not interactive mode, return false!')
            return AskFlag('no')
        _print_title(title)
        Logger.stdout(f'There are {len(options)} options available...')
        for i , option in enumerate(options):
            Logger.stdout(f'{i+1:02d}. {option}' , indent = 1)
        flag = cls.Selections(len(options) , confirm = confirm , multiple = multiple)
        if flag.yes:
            if multiple:
                flag.result = [options[i - 1] for i in flag.result]
            else:
                flag.result = options[flag.result - 1]
        return flag

    @classmethod
    def LoopTillExit(
        cls , ask = True , message : str = 'Do you want to try again?', * , 
        max_trials : int = 20
    ) -> Generator[LoopFlag, None, None]:
        """
        Loop until the user exits. Basic use:

        >> 1. if there is ask inside loop:
        for loop_flag in AskFor.LoopTillExit(True , message = f'Do you want to check more?' , max_trials = 100):
            flag = AskFor.Confirmation(title = f'Do you want to check?')
            if not loop_flag.set_flag(flag):
                continue
            # do something

        >> 2. if there is no ask inside loop:
        for _ in AskFor.LoopTillExit(True , message = f'Do you want to check more?' , max_trials = 100):
            # do something
        
        """
        if cls.not_interactive:
            Logger.error('Not interactive mode, return!')
            return
        for trial in range(max_trials):
            loop_flag = LoopFlag(round = trial)
            yield loop_flag
            if loop_flag.flag.no:
                return
            if not ask:
                continue
            retry_flag = AskFor.Retry(message)
            if retry_flag.no:
                return