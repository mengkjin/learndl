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

    def run(self) -> None:
        from src.proj.util.cli import AskFor
        from src.proj.util.cli.script_session import as_script_main
        from src.proj.util.script.param_schema import ScriptParamSchema

        runners = iter_runnable_scripts()
        if not runners:
            Logger.note('No runnable pipeline scripts found.')
            return

        label_to_runner = {runner.format_path: runner for runner in runners}
        labels = list(label_to_runner.keys())

        for loop in AskFor.LoopTillExit(message='Run another script?'):
            flag_script = AskFor.Options(
                labels,
                confirm=False,
                multiple=False,
                title='Which pipeline script to run?',
                help_description='Runnable scripts from scripts/ (numbered folders). Header YAML defines parameters.',
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
                    help_description=f'Configure parameters for [{runner.script_key}] before execution.',
                )
                if not loop.set_flag(flag_kwargs) or flag_kwargs.result is None:
                    continue

                kwargs = flag_kwargs.result
                Logger.note(f'Running [{runner.script_key}] with {kwargs}')
                main(**kwargs)

    @staticmethod
    def _load_main(runner: ScriptRunner) -> Callable[..., Any]:
        from src.proj.util.filesys.dynamic_import import dynamic_modules

        for module in dynamic_modules(runner.script):
            return module.main
        raise FileNotFoundError(f'Script main not found: {runner.script}')
