"""Direct calls related to research of this project."""

from __future__ import annotations
from collections.abc import Callable
from datetime import datetime , timedelta
from typing import Any

from src.proj import MACHINE , PATH , Logger
from src.api.util.direct_call import DirectCall
from src.proj.util.cli.script_session import as_script_main

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
        Logger.critical(f'Training schedule model list {self.schedule_names()} started')
        with as_script_main(self.SCHEDULE_SCRIPT):
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
        Logger.success('Training schedule model list completed')

class ScheduleModel(DirectCall):
    """Train a single schedule model."""
    category = 'Research'
    SCHEDULE_SCRIPT = CarryOutScheduleWorkList.SCHEDULE_SCRIPT

    def run(self) -> None:
        from src.proj import Options
        from src.proj.util.cli import AskFor
        from src.proj.util.script.param_schema import ScriptParamSchema

        schedules = Options.available_schedules(refresh = True)
        if not schedules:
            Logger.note('No schedule configs found.')
            return

        with as_script_main(self.SCHEDULE_SCRIPT):
            main = CarryOutScheduleWorkList._load_schedule_main()
            schema = ScriptParamSchema.from_script(self.SCHEDULE_SCRIPT, main=main)
            for loop in AskFor.LoopTillExit(message = 'Do you want to train another schedule model?'):
                flag_schedule = AskFor.Options(
                    schedules , confirm = False , multiple = False , allow_back = False ,
                    title = 'Which schedule model to train?',
                    help_description=(
                        'Schedule configs live under configs/model/schedule/. '
                        'Each name maps to a training plan (modules, dates, algo).'
                    ),
                    extra_help_lines=(
                        'Type / for magic commands (e.g. /help, /history).',
                    ),
                )
                if not loop.set_flag(flag_schedule) or flag_schedule.result is None:
                    continue

                schedule_name = flag_schedule.result
                Logger.note(f'Selected schedule [{schedule_name}]')

                flag_kwargs = AskFor.ScriptKwargs(
                    schema,
                    preset={'schedule_name': schedule_name},
                    help_description=(
                        f'Train schedule [{schedule_name}]. '
                        'resume: continue checkpoints; short_test: truncated run; start/end: date ints.'
                    ),
                    extra_help_lines=(
                        'Choose defaults to use YAML/resume settings, or customize each parameter.',
                    ),
                )
                if not loop.set_flag(flag_kwargs) or flag_kwargs.result is None:
                    continue

                kwargs = flag_kwargs.result
                Logger.note(f'Training schedule [{schedule_name}] with {kwargs}')
                main(**kwargs)