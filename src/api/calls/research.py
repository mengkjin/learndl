"""Direct calls related to research of this project."""

from __future__ import annotations
import sys
from collections.abc import Callable
from datetime import datetime , timedelta
from typing import Any

from src.proj import MACHINE , PATH , Logger
from src.api.util.direct_call import DirectCall

__all__ = ['CarryOutScheduleWorkList' , 'ScheduleModel']

class CarryOutScheduleWorkList(DirectCall):
    """Carry out training of a predefined schedule model list."""
    category = 'Research'
    max_test_schedules = 3
    SCHEDULE_SCRIPT = PATH.scpt.joinpath('4_train' , '2_schedule_model.py')
    @classmethod
    def get_schedules(cls , exclude_recent_created: bool = True) -> list[str]:
        ret = list(PATH.read_yaml(PATH.sched_worklist)['fit'])
        if exclude_recent_created:
            ret = [schedule for schedule in ret if not cls._is_schedule_created_recently(schedule)]
        return ret if MACHINE.platform_server else ret[:cls.max_test_schedules]
    @classmethod
    def schedule_names(cls) -> str:
        schedules = cls.get_schedules(exclude_recent_created = True)
        return ', '.join(schedules)
    @classmethod
    def get_schedule_resume_param(cls) -> bool:
        return bool(PATH.read_yaml(PATH.sched_worklist)['resume'])
    @classmethod
    def get_description(cls , **kwargs) -> str:
        return f'Carry out training of a predefined schedule model list: {cls.schedule_names()}'
    @classmethod
    def _ensure_main_script_file(cls) -> None:
        """Let BackendTaskRecorder resolve script when launched via ``python -c``."""
        main_module = sys.modules['__main__']
        cls.current_file = getattr(main_module, '__file__', None)
        main_module.__file__ = str(cls.SCHEDULE_SCRIPT.resolve())
    @classmethod
    def _restore_main_script_file(cls) -> None:
        """Restore the main script file."""
        main_module = sys.modules['__main__']
        main_module.__file__ = cls.current_file
    @classmethod
    def _load_schedule_main(cls) -> Callable[..., Any]:
        from src.proj.util.filesys.dynamic_import import dynamic_modules
        for module in dynamic_modules(cls.SCHEDULE_SCRIPT):
            return module.main
        raise FileNotFoundError(f'Schedule model script not found: {cls.SCHEDULE_SCRIPT}')
    @classmethod
    def _get_latest_creation_time(cls , schedule_name: str):
        from src.res.model.util import ModelConfig
        config = ModelConfig(schedule_name = schedule_name , vb_level = 'never')
        return config.base_path.get_creation_time(all_resumables = True)

    @classmethod
    def _is_schedule_created_recently(cls , schedule_name: str , days: int = 7):
        max_creation_time = cls._get_latest_creation_time(schedule_name)
        if max_creation_time is None:
            return False
        return max_creation_time > datetime.now() - timedelta(days = days)

    def run(self) -> None:
        self._ensure_main_script_file()
        Logger.critical(f'Training schedule model list {self.schedule_names()} started')
        try:
            for schedule_name in self.get_schedules():
                Logger.note(f'Training schedule model: {schedule_name}')
                main = self._load_schedule_main()
                main(
                    schedule_name=schedule_name,
                    short_test=None,
                    resume=self.get_schedule_resume_param(),
                    start=None,
                    end=None,
                    email=True,
                )
        finally:
            self._restore_main_script_file()
        Logger.success('Training schedule model list completed')

class ScheduleModel(DirectCall):
    """Train a single schedule model."""
    category = 'Research'
    SCHEDULE_SCRIPT = CarryOutScheduleWorkList.SCHEDULE_SCRIPT

    @classmethod
    def _parse_resume_short_test(cls , raw : str) -> tuple[bool , bool | None] | None:
        """Parse ``resume,short_test`` input; return ``None`` when invalid."""
        raw = raw.strip()
        if not raw:
            return False , None
        if ',' not in raw:
            return None

        resume_s , short_test_s = (part.strip() for part in raw.split(',' , 1))

        def _parse_bool_or_empty(part : str , default : bool | None) -> tuple[bool | None , bool]:
            if not part:
                return default , True
            lowered = part.lower()
            if lowered in ('true' , 'yes' , 'y' , '1'):
                return True , True
            if lowered in ('false' , 'no' , 'n' , '0'):
                return False , True
            return default , False

        resume , resume_ok = _parse_bool_or_empty(resume_s , False)
        if not resume_ok or not isinstance(resume , bool):
            return None
        short_test , short_test_ok = _parse_bool_or_empty(short_test_s , None)
        if not short_test_ok:
            return None
        return resume , short_test

    def run(self) -> None:
        from src.proj import Options
        from src.proj.util.functional.ask import AskFor

        schedules = Options.available_schedules(refresh = True)
        if not schedules:
            Logger.note('No schedule configs found.')
            return

        CarryOutScheduleWorkList._ensure_main_script_file()
        try:
            main = CarryOutScheduleWorkList._load_schedule_main()
            for loop in AskFor.LoopTillExit(message = 'Do you want to train another schedule model?'):
                flag_schedule = AskFor.Options(
                    schedules , confirm = False , multiple = False ,
                    title = 'Which schedule model to train?',
                )
                if not loop.set_flag(flag_schedule) or flag_schedule.result is None:
                    continue

                schedule_name = flag_schedule.result
                Logger.note(f'Selected schedule [{schedule_name}]')

                flag_params = AskFor.string(
                    title = (
                        'Override resume,short_test? '
                        '(default False,None; empty for defaults; '
                        'format: a,b where a/b empty or True/False)'
                    ),
                )
                if not loop.set_flag(flag_params):
                    continue
                assert flag_params.result is not None

                parsed = self._parse_resume_short_test(flag_params.result)
                if parsed is None:
                    Logger.error(f'Invalid resume,short_test input: {flag_params.result!r}')
                    loop.set_flag(AskFor.flag('invalid'))
                    continue

                resume , short_test = parsed
                Logger.note(f'Training schedule [{schedule_name}] with resume={resume}, short_test={short_test}')
                main(
                    schedule_name = schedule_name,
                    short_test = short_test,
                    resume = resume,
                    start = None,
                    end = None,
                    email = True,
                )
        finally:
            CarryOutScheduleWorkList._restore_main_script_file()