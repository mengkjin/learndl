"""Direct calls for running and stopping pipeline scripts."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.api.util.backend.script import ScriptRunner, iter_runnable_scripts
from src.api.util.direct_call import DirectCall
from src.proj import Logger

__all__ = ['KillRunningScript', 'RunPipelineScript']


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


class KillRunningScript(DirectCall):
    """Kill one running project script task after an explicit confirmation."""

    category = 'Basic'
    _REFRESH_LABEL = 'Refresh'
    # Listing must not call TaskItem.refresh(): a dead PID can be marked killed and emailed.
    _ALIVE_PROCESS_STATUSES = frozenset({'running', 'sleeping', 'disk-sleep'})

    @classmethod
    def get_description(cls, **kwargs) -> str:
        return (
            'List running project script tasks (PID, file name, start time) and kill a selected one. '
            'Refresh reloads the menu. Kill asks for confirmation and defaults to No.'
        )

    def run(self) -> None:
        from src.api.util.backend.task import TaskDatabase, TaskItem, timestamp
        from src.proj.util.cli import AskFor
        from src.proj.util.shell import process

        db = TaskDatabase()
        while True:
            with db.conn_handler as (_, cursor):
                cursor.execute(
                    """
                    SELECT task_id FROM task_records
                    WHERE status IN ('running', 'starting')
                    ORDER BY COALESCE(start_time, create_time), task_id
                    """
                )
                task_ids = [row['task_id'] for row in cursor.fetchall()]

            tasks: list[TaskItem] = []
            for task_id in task_ids:
                item = TaskItem.load(task_id, db)
                if item.pid is None or item.pid == os.getpid() or not item.is_running:
                    continue
                if process.check_status(item.pid) not in self._ALIVE_PROCESS_STATUSES:
                    continue
                tasks.append(item)

            if not tasks:
                Logger.note('No running script tasks to kill.')

            labels = [self._REFRESH_LABEL]
            label_to_task: dict[str, TaskItem] = {}
            option_help = {self._REFRESH_LABEL: 'Reload running script tasks and redraw this menu.'}
            for item in tasks:
                filename = Path(item.script).name
                label = f'PID {item.pid} | {filename} | {item.time_str("start")}'
                if label in label_to_task:
                    label = f'{label} | {item.id}'
                labels.append(label)
                label_to_task[label] = item
                option_help[label] = f'{item.script_key} | {item.id}'

            flag = AskFor.Options(
                labels,
                confirm=False,
                multiple=False,
                title='Kill a running script?',
                help_description=(
                    'Refresh reloads this menu. Selecting a script asks for confirmation; '
                    'the default answer is No. « Back (q) » leaves this pane.'
                ),
                option_help=option_help,
            )
            if flag.exit:
                return
            if not flag.valid or flag.result is None or flag.result == self._REFRESH_LABEL:
                continue

            selected = label_to_task[flag.result]
            filename = Path(selected.script).name
            confirm = AskFor.Confirmation(
                title=f'Kill PID {selected.pid} ({filename})?',
                help_description='Default is No. Yes terminates the process, then kills it if it is still alive.',
            )
            if not confirm.valid:
                Logger.note(f'Kill cancelled for PID {selected.pid}')
                continue

            current = TaskItem.load(selected.id, db)
            still_alive = (
                current.id == selected.id
                and current.pid == selected.pid
                and current.pid is not None
                and current.pid != os.getpid()
                and current.is_running
                and process.check_status(current.pid) in self._ALIVE_PROCESS_STATUSES
            )
            if not still_alive:
                Logger.error(f'Task [{selected.id}] is no longer the same running process; kill skipped.')
                continue
            if not current.kill():
                Logger.error(f'Failed to kill PID {current.pid} ({filename})')
                continue
            if current.is_running:
                current.update(
                    {
                        'status': 'killed',
                        'end_time': timestamp(),
                        'exit_code': 1,
                        'exit_error': 'Killed by operator from CLI',
                    },
                    sync=True,
                )
            Logger.note(f'Killed PID {selected.pid} ({filename})')
