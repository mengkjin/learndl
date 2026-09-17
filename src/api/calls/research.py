"""Direct calls related to research of this project."""

from __future__ import annotations
from collections.abc import Callable
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
    def get_schedules(cls) -> list[str]:
        from src.res.model.util.config.config import ScheduleConfig

        ret = list(PATH.read_yaml(PATH.sched_worklist)['fit'])
        missing = [name for name in ret if not ScheduleConfig.check_name_exist(name)]
        if missing:
            Logger.warning(f'Schedule configs missing from worklist (skipped): {", ".join(missing)}')
            ret = [name for name in ret if name not in set(missing)]
        return ret if MACHINE.platform_server else ret[:cls.max_test_schedules]
    @classmethod
    def schedule_names(cls) -> str:
        schedules = cls.get_schedules()
        return ', '.join(schedules) if schedules else '(none)'
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
    def _release_gpu_memory(cls , task: Any | None = None) -> None:
        """Drop retained training refs and return unused CUDA memory to the driver.

        Always safe to call after success or failure. ``empty_cache`` alone is not
        enough if ``AutoRunTask.func_return`` still holds large objects.
        """
        import gc

        import torch

        if task is not None and hasattr(task , 'func_return'):
            task.func_return = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda , 'ipc_collect'):
                torch.cuda.ipc_collect()

    @classmethod
    def _train_one_schedule(cls , main: Callable[..., Any] , **kwargs: Any) -> Any:
        """Run one schedule training and always release GPU memory afterward."""
        schedule_name = kwargs.get('schedule_name' , '?')
        Logger.note(f'Training schedule model: {schedule_name}')
        task = None
        try:
            task = main(**kwargs)
            return task
        finally:
            cls._release_gpu_memory(task)
            Logger.note(f'Schedule [{schedule_name}] finished; GPU memory released')

    def run(self) -> None:
        import portalocker

        from src.api.calls.worklist_state import AutomaticDeferred, WorklistState
        from src.res.model.util.training_history import collect_training_runs, file_revision

        worklist_revision = file_revision(PATH.sched_worklist)
        # Parse the exact bytes fingerprinted above, once per invocation.
        import yaml
        worklist = yaml.safe_load(worklist_revision['content'])
        resume, force = worklist.get('resume', False), worklist.get('force', False)
        if type(resume) is not bool or type(force) is not bool:
            raise ValueError('worklist resume and force must be YAML booleans')
        from src.res.model.util.config.config import ScheduleConfig
        schedules = list(dict.fromkeys(worklist['fit']))
        missing = [name for name in schedules if not ScheduleConfig.check_name_exist(name)]
        if missing:
            Logger.warning(f'Schedule configs missing from worklist (skipped): {", ".join(missing)}')
            schedules = [name for name in schedules if name not in set(missing)]
        if only_schedule := self.kwargs.get('only_schedule'):
            if only_schedule not in schedules:
                raise ValueError(f'Schedule [{only_schedule}] is no longer in worklist')
            schedules = [only_schedule]
        if not MACHINE.platform_server:
            schedules = schedules[:self.max_test_schedules]
        Logger.critical(f'Training schedule model list {", ".join(schedules)} started')
        failed = []
        with as_script_main(self.SCHEDULE_SCRIPT):
            main = self._load_schedule_main()
            for schedule_name in schedules:
                state = WorklistState(schedule_name)
                lock = state.lock()
                try:
                    lock.acquire()
                except portalocker.exceptions.LockException:
                    if self.kwargs.get('automatic'):
                        raise AutomaticDeferred('Schedule lock occupied')
                    Logger.note(f'Skip [{schedule_name}]: another worklist process is running it')
                    continue
                try:
                    previous = state.read()
                    revisions = state.revisions(schedule_name)
                    revisions['worklist'] = worklist_revision
                    automatic = bool(self.kwargs.get('automatic', False))
                    if automatic and self.kwargs.get('expected_version'):
                        from src.api.task_monitor.scheduling.idle_worklist import version
                        if version(revisions) != self.kwargs['expected_version']:
                            raise AutomaticDeferred('Configuration changed before training')
                    effective_force = force and not (automatic and previous and previous.get('status') == 'success' and previous.get('revisions') == revisions)
                    reason, effective_resume = state.decision(previous, revisions, force=effective_force, resume=resume)
                    if reason == 'completed':
                        if automatic:
                            raise AutomaticDeferred('Already completed')
                        Logger.note(f'Skip [{schedule_name}]: successful training already recorded')
                        continue
                    Logger.note(f'Run [{schedule_name}]: {reason}; resume={effective_resume}')
                    model_path = previous.get('model_path') if previous else None
                    details: dict[str, Any] = {'revisions': revisions, 'reason': reason, 'resume': effective_resume,
                               'force': force, 'model_path': model_path, 'training_run_ids': []}
                    state.save(status='running', **details)
                    def record_progress(run):
                        # Persist the chosen directory before training, including hard-kill recovery.
                        if run['run_id'] not in details['training_run_ids']:
                            details['training_run_ids'].append(run['run_id'])
                        if run.get('model_path') or not effective_resume:
                            details['model_path'] = run.get('model_path')
                        state.save(status='running', **details)

                    with collect_training_runs(on_update=record_progress) as runs:
                        try:
                            kwargs = {'base_path': model_path} if effective_resume and model_path else {}
                            task = self._train_one_schedule(
                                main, schedule_name=schedule_name, short_test=None,
                                resume=effective_resume, start=None, end=None, email=True,
                                resume_selection='latest' if automatic else 'interactive', **kwargs,
                            )
                            if not (task.success and task.execution_success and runs
                                    and all(run['status'] == 'success' for run in runs)):
                                raise RuntimeError(f'Schedule [{schedule_name}] did not complete successfully')
                            # A pull/edit during training must not certify the new configuration.
                            if state.revisions(schedule_name) != revisions:
                                raise RuntimeError('Worklist or schedule changed during training; completion not cached')
                        except BaseException as exc:
                            details['training_run_ids'] = [run['run_id'] for run in runs]
                            if runs and runs[-1].get('model_path'):
                                details['model_path'] = runs[-1]['model_path']
                            state.save(status='failed', error=f'{type(exc).__name__}: {exc}', **details)
                            if not isinstance(exc, Exception):
                                raise
                            failed.append(schedule_name)
                            Logger.warning(str(exc))
                        else:
                            details['model_path'] = runs[-1]['model_path']
                            details['training_run_ids'] = [run['run_id'] for run in runs]
                            state.save(status='success', **details)
                finally:
                    lock.release()
        if failed:
            raise RuntimeError(f'Training failed for schedules: {", ".join(failed)}')
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
                CarryOutScheduleWorkList._train_one_schedule(main , **kwargs)
