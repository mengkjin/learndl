"""Direct calls related to research of this project."""

from __future__ import annotations

from src.proj import Logger
from src.api.util.direct_call import DirectCall
from src.proj.util.cli.script_session import as_script_main


__all__ = ['CarryOutScheduleWorkList' , 'ScheduleModel']

from src.res.model.workflow.execution import ScheduleWorklist

class CarryOutScheduleWorkList(ScheduleWorklist, DirectCall):
    """Interactive entry for the shared schedule worklist service."""
    category = 'Research'

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
            main = None
            schema = ScriptParamSchema.from_source(self.SCHEDULE_SCRIPT)
            if schema is None:
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
                if main is None:
                    main = CarryOutScheduleWorkList._load_schedule_main()
                from src.automation.scheduling.training_maintenance import TrainingPaused
                if isinstance(CarryOutScheduleWorkList._train_one_schedule(main, **kwargs), TrainingPaused):
                    return
