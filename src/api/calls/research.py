"""Direct calls related to research of this project."""

from __future__ import annotations
import sys
from collections.abc import Callable
from typing import Any

from src.proj import MACHINE , PATH , Logger
from src.api.util import DirectCall

__all__ = ['CarryOutScheduleModelList' , 'CheckAllConfigFiles']

class CarryOutScheduleModelList(DirectCall):
    """Carry out training of a predefined schedule model list."""
    category = 'Research'
    schedules : tuple[str, ...] = (
        'gru_30m_new',
        'gru_dfl2_new',
        'gru_dfl2cs_new',
    )
    max_test_schedules = 1
    SCHEDULE_SCRIPT = PATH.scpt.joinpath('4_train' , '2_schedule_model.py')
    @classmethod
    def get_schedules(cls) -> list[str]:
        ret = list(cls.schedules)
        return ret if MACHINE.platform_server else ret[:cls.max_test_schedules]
    @classmethod
    def schedule_names(cls) -> str:
        return ', '.join(cls.get_schedules())
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
        finally:
            self._restore_main_script_file()
        Logger.success('Training schedule model list completed')

class CheckAllConfigFiles(DirectCall):
    """Check and auto modify all config files."""
    category = 'Research'
    def run(self) -> None:
        from src.res.model.util.config.inspector import ModelConfigsInspector
        from src.res.model.util.config.modifier import ModelConfigsBatchModifier
        from src.proj import Logger
        Logger.stdout('Checking all config files...')
        modifier = ModelConfigsBatchModifier()
        modifier.batch_modify()
        inspecter = ModelConfigsInspector()
        inspecter.inspect_key_values()
        Logger.success('All config files checked.')
