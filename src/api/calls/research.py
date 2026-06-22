"""Direct calls related to research of this project."""

from __future__ import annotations
import sys
from collections.abc import Callable
from datetime import datetime , timedelta
from typing import Any

from src.proj import MACHINE , PATH , Logger
from src.api.util.direct_call import DirectCall

__all__ = ['CarryOutScheduleWorkList']

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
    def update_worklist(cls , finish : str) -> None:
        content = PATH.read_yaml(PATH.sched_worklist)
        content['fit'].remove(finish)
        PATH.dump_yaml(content , PATH.sched_worklist , overwrite = True)
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
                    resume=False,
                    start=None,
                    end=None,
                    email=True,
                )
                self.update_worklist(schedule_name)
        finally:
            self._restore_main_script_file()
        Logger.success('Training schedule model list completed')

