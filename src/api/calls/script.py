"""Direct calls for running pipeline scripts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.api.util.backend.script import ScriptRunner, iter_runnable_scripts
from src.api.util.direct_call import DirectCall
from src.proj import Logger

__all__ = ['RunPipelineScript']


class RunPipelineScript(DirectCall):
    """Interactively run a pipeline script from scripts/."""

    category = 'Basic'

    def __init__(self, script_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if script_key is not None:
            self.kwargs['script_key'] = script_key

    @property
    def script_key(self) -> str | None:
        return self.kwargs.get('script_key')

    @classmethod
    def get_description(cls, script_key: str | None = None, **kwargs) -> str:
        if script_key:
            return f'Run pipeline script [{script_key}] with interactive parameter prompts.'
        return 'Interactively pick and run a pipeline script from scripts/.'

    def run(self) -> None:
        from src.proj.util.cli import AskFor
        from src.proj.util.cli.script_session import as_script_main
        from src.proj.util.script.param_schema import ScriptParamSchema

        fixed_key = self.script_key
        runners = iter_runnable_scripts()
        if not runners:
            Logger.note('No runnable pipeline scripts found.')
            return

        if fixed_key is not None and not any(runner.script_key == fixed_key for runner in runners):
            Logger.note(f'Script [{fixed_key}] is not runnable on this machine.')
            return

        label_to_runner = {runner.format_path: runner for runner in runners}
        labels = list(label_to_runner.keys())
        loop_message = 'Run again with different parameters?' if fixed_key else 'Run another script?'

        for loop in AskFor.LoopTillExit(message=loop_message):
            if fixed_key:
                runner = ScriptRunner.from_key(fixed_key)
            else:
                flag_script = AskFor.Options(
                    labels,
                    confirm=False,
                    multiple=False,
                    allow_back=False,
                    title='Which pipeline script to run?',
                    help_description=(
                        'Runnable scripts from scripts/ (numbered folders). '
                        'Header YAML defines parameters; you will configure them next.'
                    ),
                )
                if not loop.set_flag(flag_script) or flag_script.result is None:
                    continue
                runner = label_to_runner[flag_script.result]

            Logger.note(f'Selected script [{runner.script_key}]')

            main = self._load_main(runner)
            with as_script_main(runner.script):
                schema = ScriptParamSchema.from_script(runner.script, main=main)
                flag_kwargs = AskFor.ScriptKwargs(
                    schema,
                    help_description=(
                        f'Configure parameters for [{runner.script_key}] before execution. '
                        'Required fields are prompted first; then accept defaults or enter each remaining field in order.'
                    ),
                )
                if not loop.set_flag(flag_kwargs) or flag_kwargs.result is None:
                    continue

                kwargs = flag_kwargs.result
                if runner.header.email:
                    kwargs = {**kwargs, 'email': True}
                Logger.note(f'Running [{runner.script_key}] with {kwargs}')
                main(**kwargs)

    @staticmethod
    def _load_main(runner: ScriptRunner) -> Callable[..., Any]:
        from src.proj.util.filesys.dynamic_import import dynamic_modules

        for module in dynamic_modules(runner.script):
            return module.main
        raise FileNotFoundError(f'Script main not found: {runner.script}')
