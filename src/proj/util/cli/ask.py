"""Interactive terminal prompts for DirectCall and CLI workflows."""
from __future__ import annotations

import sys
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Generic, Literal, Self, TypeAlias, TypeVar

import questionary
from IPython.core.getipython import get_ipython

from src.proj.util.cli.help_context import AskHelpContext, build_option_help_map, set_ask_help_context
from src.proj.util.cli.prompts import prompt_checkbox, prompt_confirm, prompt_select, prompt_text
from src.proj.util.cli.session import GitHeadWatcher, ProcessReload
from src.proj.log import Logger

__all__ = ['AskFlag', 'AskFlagType', 'AskFor', 'AskHelpContext', 'LoopFlag', 'ExitFlags', 'EXIT_MENU_VALUE']

AskFlagType: TypeAlias = Literal['valid', 'invalid', 'exit']
ExitFlags: frozenset[str] = frozenset(['q'])
EXIT_MENU_VALUE = '__ask__:exit'
EXIT_MENU_LABEL = '« Back (q) »'
T = TypeVar('T')


class AskFlag(Generic[T]):
    """Result wrapper for :class:`AskFor` prompts."""

    def __init__(self, flag: AskFlagType):
        self._flag: AskFlagType = flag

    def __repr__(self) -> str:
        return f'AskFlag({self._flag})'

    def __str__(self) -> str:
        return self._flag

    def __bool__(self) -> bool:
        return self.valid

    def set_flag(self, flag: AskFlagType) -> Self:
        self._flag = flag
        return self

    def set_result(self, results: list[T]) -> Self:
        self._results: list[T] = results
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
        assert len(self.results) <= 1, 'result must be a single value'
        return self.results[0] if self.results else None

    @property
    def results(self) -> list[T]:
        if not hasattr(self, '_results'):
            return []
        return self._results


class LoopFlag:
    """Loop state for :meth:`AskFor.LoopTillExit`."""

    def __init__(self, round: int = 0, flag: AskFlagType = 'valid'):
        self.round: int = round
        self._flag: AskFlag = AskFlag(flag=flag)

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

    def set_flag(self, flag: AskFlag | None) -> LoopFlag:
        if flag is None:
            return self
        self._flag = flag
        return self

    @property
    def flag(self) -> AskFlag:
        return self._flag


class AskFor:
    """Interactive prompts backed by questionary."""

    USE_CHECKBOX_THRESHOLD = 10

    @classmethod
    def _append_exit_choice(
        cls,
        choices: list[questionary.Choice],
        *,
        allow_back: bool,
    ) -> list[questionary.Choice]:
        if not allow_back:
            return choices
        return [
            *choices,
            questionary.Choice(title=EXIT_MENU_LABEL, value=EXIT_MENU_VALUE),
        ]

    @classmethod
    def _is_exit_menu_value(cls, value: Any) -> bool:
        return value == EXIT_MENU_VALUE

    @classmethod
    def _resolve_use_checkbox(cls, option_count: int, use_checkbox: bool | None) -> bool:
        if use_checkbox is not None:
            return use_checkbox
        return option_count < cls.USE_CHECKBOX_THRESHOLD

    @classmethod
    def flag(cls, flag: AskFlagType = 'valid') -> AskFlag:
        return AskFlag(flag=flag)

    @classmethod
    def check_interactive(cls) -> bool:
        """Check if the current environment is interactive."""
        if sys.stdin.isatty():
            return True
        try:
            shell = get_ipython().__class__.__name__
            return shell in {'ZMQInteractiveShell', 'TerminalInteractiveShell'}
        except NameError:
            Logger.error('Not interactive mode, return!')
            return False

    @classmethod
    def print_title(cls, title: str) -> None:
        """Print the ask title."""
        if title:
            Logger.stdout(f'{title}', color='lightpurple')

    @classmethod
    @contextmanager
    def _help_scope(
        cls,
        *,
        title: str = '',
        help_description: str = '',
        options: Sequence[Any] | None = None,
        option_help: Mapping[Any, str] | Sequence[str] | None = None,
        extra_lines: Sequence[str] = (),
    ):
        if not help_description and not option_help and not extra_lines:
            yield
            return
        option_list = list(options or [])
        set_ask_help_context(
            AskHelpContext(
                prompt_title=title,
                description=help_description,
                options=option_list,
                option_help=build_option_help_map(option_list, option_help),
                extra_lines=tuple(extra_lines),
            ),
        )
        try:
            yield
        finally:
            set_ask_help_context(None)

    @classmethod
    def _build_option_labels(
        cls,
        options: int | list[Any],
        *,
        start_index: int,
    ) -> tuple[list[str], int]:
        if isinstance(options, int):
            num = options
            option_strs = [f'#{i + start_index:02d}' for i in range(num)]
        else:
            num = len(options)
            option_strs = [f'{i + start_index}.{option}' for i, option in enumerate(options)]
        return option_strs, num

    @classmethod
    def _build_index_choices(
        cls,
        options: int | list[Any],
        *,
        start_index: int,
    ) -> tuple[list[questionary.Choice], list[str], int]:
        option_strs, num = cls._build_option_labels(options, start_index=start_index)
        choices = [
            questionary.Choice(title=label, value=i + start_index)
            for i, label in enumerate(option_strs)
        ]
        return choices, option_strs, num

    @classmethod
    def _selections_from_text(
        cls,
        *,
        confirm: bool,
        multiple: bool,
        start_index: int,
        option_strs: list[str],
        num: int,
    ) -> AskFlag[int]:
        min_index, max_index = start_index, num + start_index - 1
        if multiple:
            while True:
                selection = prompt_text(
                    f'Choose from {min_index} to {max_index}, (sep by "," or range by "-" , q to quit)',
                )
                if selection is None:
                    return AskFlag('exit')
                if selection.lower() in ExitFlags:
                    return AskFlag('exit')
                break
            if not selection.strip().replace('-', '').replace(',', '').replace(' ', '').isdigit():
                Logger.error(f'Contains non-digit characters: {selection}')
                return AskFlag('invalid')
            if '-' in selection:
                start, end = selection.split('-', 1)
                choices = list(range(int(start.strip()), int(end.strip()) + 1))
            elif ',' in selection:
                choices = [int(s.strip()) for s in selection.split(',') if s.strip()]
            else:
                choices = [int(selection.strip())]
            choices = [int(i) for i in choices]
        else:
            while True:
                selection = prompt_text(f'Choose from {min_index} to {max_index} (q to quit)')
                if selection is None:
                    return AskFlag('exit')
                if selection.lower() in ExitFlags:
                    return AskFlag('exit')
                break
            if not selection.isdigit():
                Logger.error(
                    f'Invalid input: {selection} , please choose from {min_index} to {max_index} or q to quit',
                )
                return AskFlag('invalid')
            choices = [int(selection)]
        if any(s < start_index or s > max_index for s in choices):
            Logger.error(f'Contains indices out of range [{min_index}-{max_index}]: {selection}')
            return AskFlag('invalid')
        if confirm:
            message = (
                f'Are you sure to select {option_strs[choices[0] - start_index] if len(choices) == 1 else choices}?'
            )
            flag = cls.Confirmation(title=message)
            return AskFlag('valid').set_result(choices) if flag.valid else AskFlag('invalid')
        return AskFlag('valid').set_result(choices)

    @classmethod
    def _selections_from_checkbox(
        cls,
        *,
        confirm: bool,
        multiple: bool,
        start_index: int,
        choices: list[questionary.Choice],
        option_strs: list[str],
        allow_back: bool = True,
    ) -> AskFlag[int]:
        menu_choices = cls._append_exit_choice(choices, allow_back=allow_back)
        if multiple:
            selected = prompt_checkbox('Choose one or more options', choices=menu_choices)
            if selected is None:
                return AskFlag('exit')
            if any(cls._is_exit_menu_value(value) for value in selected):
                return AskFlag('exit')
            if not selected:
                Logger.error('No options selected')
                return AskFlag('invalid')
            selected_indices = sorted(int(i) for i in selected)
        else:
            selected_index = prompt_select('Choose an option', choices=menu_choices)
            if selected_index is None:
                return AskFlag('exit')
            if cls._is_exit_menu_value(selected_index):
                return AskFlag('exit')
            selected_indices = [int(selected_index)]
        min_index, max_index = start_index, start_index + len(choices) - 1
        if any(i < min_index or i > max_index for i in selected_indices):
            Logger.error(f'Contains indices out of range [{min_index}-{max_index}]: {selected_indices}')
            return AskFlag('invalid')
        if confirm:
            message = (
                f'Are you sure to select {option_strs[selected_indices[0] - start_index] if len(selected_indices) == 1 else selected_indices}?'
            )
            flag = cls.Confirmation(title=message)
            return AskFlag('valid').set_result(selected_indices) if flag.valid else AskFlag('invalid')
        return AskFlag('valid').set_result(selected_indices)

    @classmethod
    def String(
        cls,
        title: str = '',
        *,
        help_description: str = '',
        extra_help_lines: Sequence[str] = (),
    ) -> AskFlag[str]:
        if not cls.check_interactive():
            return AskFlag('exit')
        with cls._help_scope(
            title=title,
            help_description=help_description,
            extra_lines=extra_help_lines,
        ):
            cls.print_title(title)
            selection = prompt_text('Please input (q to quit)')
        if selection is None:
            return AskFlag('exit')
        if selection.lower() in ExitFlags:
            return AskFlag('invalid')
        return AskFlag('valid').set_result([selection])

    @classmethod
    def Confirmation(
        cls, 
        timeout: int = -1, ask_times: int = 1, title: str = '' , * , 
        help_description: str = '', extra_help_lines: Sequence[str] = ()
    ) -> AskFlag[None]:
        assert ask_times > 0, f'ask_times must be greater than 0 , but got {ask_times}'
        if timeout > 0:
            Logger.alert1('Confirmation timeout is not supported with questionary; ignoring timeout.')
        if not cls.check_interactive():
            return AskFlag('exit')
        with cls._help_scope(
            title=title,
            help_description=help_description,
            extra_lines=extra_help_lines,
        ):
            cls.print_title(title)
            for i in range(ask_times):
                message = 'Please confirm'
                if ask_times > 1:
                    message += f' ({i + 1}/{ask_times} rounds)'
                confirmed = prompt_confirm(message)
                if confirmed is None:
                    return AskFlag('exit')
                if not confirmed:
                    Logger.error(f'Confirmation is rejected at the {i + 1}th round')
                    return AskFlag('exit')
        return AskFlag('valid')

    @classmethod
    def Selections(
        cls,
        options: int | list[Any],
        confirm: bool = True,
        multiple: bool = False,
        title: str = '',
        start_index: int = 1,
        use_checkbox: bool | None = None,
        *,
        allow_back: bool = True,
        help_description: str = '',
        option_help: Mapping[Any, str] | Sequence[str] | None = None,
        extra_help_lines: Sequence[str] = (),
    ) -> AskFlag[int]:
        if not cls.check_interactive():
            return AskFlag('exit')
        if not options:
            Logger.alert1('No options provided')
            return AskFlag('invalid')
        option_values = list(range(options)) if isinstance(options, int) else list(options)
        with cls._help_scope(
            title=title,
            help_description=help_description,
            options=option_values,
            option_help=option_help,
            extra_lines=extra_help_lines,
        ):
            cls.print_title(title)
            option_strs, num = cls._build_option_labels(options, start_index=start_index)
            use_menu = cls._resolve_use_checkbox(num, use_checkbox)
            if use_menu:
                choices, option_strs, _num = cls._build_index_choices(options, start_index=start_index)
                return cls._selections_from_checkbox(
                    confirm=confirm,
                    multiple=multiple,
                    start_index=start_index,
                    choices=choices,
                    option_strs=option_strs,
                    allow_back=allow_back,
                )
            return cls._selections_from_text(
                confirm=confirm,
                multiple=multiple,
                start_index=start_index,
                option_strs=option_strs,
                num=num,
            )

    @classmethod
    def Retry(cls, title: str = '') -> AskFlag:
        if not cls.check_interactive():
            return AskFlag('exit')
        cls.print_title(title)
        choices = [
            questionary.Choice('Yes', value='y'),
            questionary.Choice('No', value='n'),
            questionary.Choice('Quit', value='q'),
        ]
        while True:
            value = prompt_select(title or 'Continue?', choices=choices)
            if value is None or value == 'q':
                return AskFlag('exit')
            if value == 'n':
                return AskFlag('invalid')
            if value == 'y':
                return AskFlag('valid')
            Logger.error(f'Invalid input: {value}')

    @classmethod
    def Options(
        cls,
        options: list[T],
        confirm: bool = True,
        multiple: bool = False,
        title: str = '',
        print_options: bool = True,
        use_checkbox: bool | None = None,
        allow_back: bool = True,
        *,
        help_description: str = '',
        option_help: Mapping[Any, str] | Sequence[str] | None = None,
        extra_help_lines: Sequence[str] = (),
    ) -> AskFlag[T]:
        if not cls.check_interactive():
            return AskFlag('exit')
        with cls._help_scope(
            title=title,
            help_description=help_description,
            options=options,
            option_help=option_help,
            extra_lines=extra_help_lines,
        ):
            cls.print_title(title)
            use_menu = cls._resolve_use_checkbox(len(options), use_checkbox)
            if print_options and not use_menu:
                Logger.stdout(f'There are {len(options)} options available...')
                for i, option in enumerate(options):
                    Logger.stdout(f'{i + 1:02d}. {option}', indent=1)
            flag = cls.Selections(
                options,
                confirm=confirm,
                multiple=multiple,
                use_checkbox=use_menu,
                allow_back=allow_back,
            )
        new_flag = AskFlag(flag._flag)
        if flag.valid:
            new_flag.set_result([options[i - 1] for i in flag.results])
        return new_flag

    @classmethod
    def ScriptKwargs(
        cls,
        schema: Any,
        *,
        preset: dict[str, Any] | None = None,
        skip: frozenset[str] = frozenset(),
        help_description: str = '',
        extra_help_lines: tuple[str, ...] = (),
    ) -> AskFlag[dict[str, Any]]:
        """Collect script kwargs via default-or-customize CLI flow."""
        from src.proj.util.cli.script_params import prompt_script_kwargs
        return prompt_script_kwargs(
            schema,
            preset=preset,
            skip=skip,
            help_description=help_description,
            extra_help_lines=extra_help_lines,
        )

    @classmethod
    def LoopTillExit(
        cls,
        ask: bool = True,
        message: str = 'Do you want to try again?',
        *,
        max_trials: int = 20,
        watch_git: bool = True,
    ) -> Generator[LoopFlag, None, None]:
        """Loop until the user exits or git HEAD changes."""
        if not cls.check_interactive():
            return
        watcher = GitHeadWatcher() if watch_git else None
        for trial in range(max_trials):
            if watcher is not None and watcher.changed():
                raise ProcessReload('git HEAD changed')
            loop_flag = LoopFlag(round=trial)
            yield loop_flag
            if loop_flag.break_loop:
                break
            if ask and AskFor.Retry(message).exit:
                break
