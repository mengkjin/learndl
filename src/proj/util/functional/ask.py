"""Process discovery, power profile (non-Windows), and CLI arg parsing helpers."""
from __future__ import annotations

import sys
from IPython.core.getipython import get_ipython
from typing import Any , Literal , TypeAlias , Generic , TypeVar , Self
from collections.abc import Generator

from src.proj.log import Logger

__all__ = ['AskFor']

AskFlagType : TypeAlias = Literal['valid' , 'invalid' , 'exit']
ExitFlags : frozenset[str] = frozenset(['q'])
T = TypeVar('T')

class AskFlag(Generic[T]):
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
    ):
        self._flag : AskFlagType = flag

    def __repr__(self) -> str:
        return f'AskFlag({self._flag})'
    def __str__(self) -> str:
        return self._flag
    def __bool__(self) -> bool:
        return self.valid

    def set_flag(self , flag : AskFlagType) -> Self:
        self._flag = flag
        return self

    def set_result(self , results : list[T]) -> Self:
        self._results : list[T] = results
        return self

    @property
    def flag(self) -> AskFlagType:
        return self._flag

    @property
    def valid(self) -> bool:
        return self._flag == 'valid'
    @property
    def exit(self) -> bool:
        return self._flag == 'exit'
    @property
    def invalid(self) -> bool:
        return self._flag == 'invalid'
    @property
    def result(self) -> T | None:
        assert len(self.results) <= 1 , f'result must be a single value'
        return self.results[0] if self.results else None
    @property
    def results(self) -> list[T]:
        if not hasattr(self , '_results'):
            return []
        return self._results

class LoopFlag:
    """
    Loop flag for the loop until exit.
    """
    def __init__(self , round : int = 0 , flag : AskFlagType = 'valid'):
        self.round : int = round
        self._flag : AskFlag = AskFlag(flag = flag)

    def __repr__(self) -> str:
        return f'LoopFlag({self.round},{self._flag})'
    def __str__(self) -> str:
        return f'Round {self.round} , Flag: {self._flag}'

    def __bool__(self) -> bool:
        return self._flag.valid

    @property
    def break_loop(self) -> bool:
        return self._flag.exit

    @property
    def continue_loop(self) -> bool:
        return self._flag.valid or self._flag.invalid

    def set_flag(self , flag : AskFlag | None) -> LoopFlag:
        if flag is None:
            return self
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

    @classmethod
    def flag(cls , flag : AskFlagType = 'valid') -> AskFlag:
        return AskFlag(flag = flag)

    @classmethod
    def check_interactive(cls) -> bool:
        """Check if the current environment is interactive."""
        if sys.stdin.isatty():
            return True
        try:
            shell = get_ipython().__class__.__name__
            return shell in {"ZMQInteractiveShell", "TerminalInteractiveShell"}
        except NameError:
            Logger.error('Not interactive mode, return!')
            return False

    @classmethod
    def print_title(cls , title : str):
        """Print the ask title."""
        if title:
            Logger.stdout(f'{title}' , color = 'lightpurple')

    @classmethod
    def string(cls , title : str = '') -> AskFlag[str]:
        if not cls.check_interactive():
            return AskFlag('exit')
        cls.print_title(title)
        selection = input(f'Please input (q to quit): ')
        if selection.lower() in ExitFlags:
            return AskFlag('invalid')
        else:
            return AskFlag('valid').set_result([selection])
        
    @classmethod
    def Confirmation(cls , timeout = -1 , ask_times = 1 , title = '') -> AskFlag[None]:
        """Prompt up to ``recurrent`` times with optional per-prompt timeout.

        Returns:
            Tuple of (inputs list, bool list from ``proceed_condition``).
        """
        assert ask_times > 0 , f'ask_times must be greater than 0 , but got {ask_times}'
        
        if not cls.check_interactive():
            return AskFlag('exit')
        cls.print_title(title)
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
                return AskFlag('exit')
            if value.strip().lower() != 'y':
                Logger.error(f'Invalid input: {value} , confirmation is rejected at the {i+1}th round')
                return AskFlag('exit') 
        return AskFlag('valid')

    @classmethod
    def Selections(
        cls , options : int | list[Any] , confirm : bool = True , 
        multiple : bool = False , title : str = '' , start_index : int = 1 , 
    ) -> AskFlag[int]:
        """Ask for selections out of a number of options starting from a given index."""
        if not cls.check_interactive():
            return AskFlag('exit')
        if not options:
            Logger.alert1('No options provided')
            return AskFlag('invalid')
        cls.print_title(title)
        if isinstance(options , int):
            num = options
            option_strs = [f'#{i+start_index:02d}' for i in range(num)]
        else:
            num = len(options)
            option_strs = [f'{i+start_index}.{option}' for i, option in enumerate(options)]
        min , max = start_index , num + start_index - 1
        if multiple:
            selection = input(f'Choose from {min} to {max}, (sep by "," or range by "-" , q to quit): ')
            if selection.lower() in ExitFlags:
                return AskFlag('exit')
            if not selection.strip().replace('-', '').replace(',', '').replace(' ', '').isdigit():
                Logger.error(f'Contains non-digit characters: {selection}')
                return AskFlag('invalid')
            if '-' in selection:
                start , end = selection.split('-')
                start , end = int(start.strip()) , int(end.strip())
                choices = list(range(start , end + 1))
            elif ',' in selection:
                choices = [int(s.strip()) for s in selection.split(',') if s.strip()]
            else:
                choices = [int(selection.strip())]

            choices = [int(i) for i in choices]
            if any(s < start_index or s > num + start_index - 1 for s in choices):
                Logger.error(f'Contains indices out of range [{min}-{max}]: {selection}')
                return AskFlag('invalid')
        else:
            selection = input(f'Choose from {min} to {max} (q to quit): ')
            if selection.lower() in ExitFlags:
                return AskFlag('exit')
            if not selection.isdigit():
                Logger.error(f'Invalid input: {selection} , please choose from {min} to {max} or q to quit')
                return AskFlag('invalid')
            choices = [int(selection)]
            if choices[0] < start_index or choices[0] > num + start_index - 1:
                Logger.error(f'Contains indices out of range [{min}-{max}]: {selection}')
                return AskFlag('invalid')

        if confirm:
            message = f'Are you sure to select {option_strs[choices[0] - start_index] if len(choices) == 1 else choices}?'
            flag = cls.Confirmation(title = message)
            return AskFlag('valid').set_result(choices) if flag.valid else AskFlag('invalid')
        else:
            return AskFlag('valid').set_result(choices)

    @classmethod
    def Retry(cls , title : str = '') -> AskFlag:
        """Ask for exit."""
        if not cls.check_interactive():
            return AskFlag('exit')
        cls.print_title(title)
        while True:
            value = input(f'Choose yes or no or quit (y/n/q): ')
            stripped_value = value.strip().lower()
            if stripped_value in ExitFlags:
                return AskFlag('exit')
            elif stripped_value == 'n':
                return AskFlag('invalid')
            elif stripped_value == 'y':
                return AskFlag('valid')
            else:
                Logger.error(f'Invalid input: {value} , please choose yes or no or quit (y/n/q)')

    @classmethod
    def Options(
        cls , options : list[T] , confirm : bool = True , 
        multiple : bool = False , title : str = '' , print_options : bool = True
    ) -> AskFlag[T]:
        """Ask for options from a list of options."""
        if not cls.check_interactive():
            return AskFlag('exit')
        cls.print_title(title)
        if print_options:
            Logger.stdout(f'There are {len(options)} options available...')
            for i , option in enumerate(options):
                Logger.stdout(f'{i+1:02d}. {option}' , indent = 1)
        flag = cls.Selections(options , confirm = confirm , multiple = multiple)
        new_flag = AskFlag(flag._flag)
        if flag.valid:
            new_flag.set_result([options[i - 1] for i in flag.results])
        return new_flag

    @classmethod
    def LoopTillExit(
        cls , ask = True , message : str = 'Do you want to try again?', * , 
        max_trials : int = 20
    ) -> Generator[LoopFlag, None, None]:
        """
        Loop until the user exits. Basic use:

        >> 1. if there is ask inside loop:
        for loop in AskFor.LoopTillExit(True , message = f'Do you want to check more?' , max_trials = 100):
            flag = AskFor.Confirmation(title = f'Do you want to check?')
            if not loop.set_flag(flag):
                continue
            # do something

        >> 2. if there is no ask inside loop:
        for _ in AskFor.LoopTillExit(True , message = f'Do you want to check more?' , max_trials = 100):
            do_something()
        
        """
        if not cls.check_interactive():
            return
        for trial in range(max_trials):
            loop_flag = LoopFlag(round = trial)
            yield loop_flag
            if loop_flag.break_loop:
                break
            if ask and AskFor.Retry(message).exit:
                break