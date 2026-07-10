"""DirectCall launcher hub."""

from __future__ import annotations

from typing import Type

from src.api.util.backend.script import ScriptRunner, iter_runnable_scripts
from src.api.util.direct_call import DirectCall
from src.proj import Logger

__all__ = ['DirectCallHub']

_TOP_LEVEL_LABELS = (
    'Launch Streamlit App',
    'Non-Research Operations',
    'Research Operations',
    'Run Pipeline Script',
)


class DirectCallHub(DirectCall):
    """Interactive hub with grouped menus; spawn DirectCalls or pipeline scripts in new panes."""

    category = 'Basic'

    @classmethod
    def get_description(cls, **kwargs) -> str:
        return (
            'CLI launcher with four top-level groups: Streamlit app, non-research tools, '
            'research workflows, and pipeline scripts. Submenus spawn the selected action in a new pane.'
        )

    @classmethod
    def _top_level_help(cls) -> dict[str, str]:
        return {
            'Git Pull': 'Pull the latest code from remote. Clear changes before pulling.',
            'Launch Streamlit App': 'Open the Streamlit interactive app in a new pane.',
            'Non-Research Operations': 'Submenu: git pull, tests, lint, preview, and project auto-fix.',
            'Research Operations': 'Submenu: data rebuild, model archive, TensorBoard, Optuna, and schedule work list.',
            'Run Pipeline Script': 'Submenu: pick a numbered script from scripts/ to run in a new pane.',
        }

    @classmethod
    def _source_code_entries(cls) -> list[tuple[str, Type[DirectCall], str]]:
        from src.api.calls.files import ProjectAutoFix
        from src.api.calls.preview import PreviewProjectFile
        from src.api.calls.source_code import CheckCodeIssues
        from src.api.calls.test import TestCode

        return [
            (
                'Test Code',
                TestCode,
                'Smoke tests for logging, quick training, and parallel factor calculation.',
            ),
            (
                'Preview Project File',
                PreviewProjectFile,
                PreviewProjectFile.get_description(),
            ),
            (
                'Check Code Issues',
                CheckCodeIssues,
                'Static scans for style, typing, and project-specific code hygiene.',
            ),
            (
                'Project AutoFix',
                ProjectAutoFix,
                'Config fixes, WezTerm template sync, and outdated log cleanup.',
            ),
        ]

    @classmethod
    def _research_entries(cls) -> list[tuple[str, Type[DirectCall], str]]:
        from src.api.calls.dashboard import OptunaDashboard, Tensorboard
        from src.api.calls.data import ReconstructPreprocessedData, RecalculateHistNorm
        from src.api.calls.files import ModelArchiveOperations
        from src.api.calls.research import CarryOutScheduleWorkList

        return [
            (
                'Reconstruct Preprocessed Data',
                ReconstructPreprocessedData,
                'Rebuild fit/predict data frames for a registered PrePros key.',
            ),
            (
                'Recalculate HistNorm',
                RecalculateHistNorm,
                'Recalculate the historical normalisation statistics.',
            ),
            (
                'Model Archive Operations',
                ModelArchiveOperations,
                'Archive, resume, rename, or inspect trained model directories.',
            ),
            (
                'Tensorboard',
                Tensorboard,
                'Browse training logs via TensorBoard in a new pane.',
            ),
            (
                'Optuna Dashboard',
                OptunaDashboard,
                'Browse hyperparameter study SQLite databases via Optuna Dashboard.',
            ),
            (
                'Carry Out Schedules',
                CarryOutScheduleWorkList,
                CarryOutScheduleWorkList.get_description(),
            ),
        ]

    @classmethod
    def _pipeline_script_entries(cls) -> list[tuple[str, ScriptRunner, str]]:
        return [
            (
                runner.format_path,
                runner,
                f'Run [{runner.script_key}] — {runner.desc or runner.script_name}.',
            )
            for runner in iter_runnable_scripts()
        ]

    @classmethod
    def _pick_direct_call(
        cls,
        entries: list[tuple[str, Type[DirectCall], str]],
        *,
        title: str,
        help_description: str,
    ) -> Type[DirectCall] | None:
        from src.proj.util.cli import AskFor

        if not entries:
            Logger.note('No actions available in this group.')
            return None

        labels = [label for label, _, _ in entries]
        label_to_cls = {label: direct_call_cls for label, direct_call_cls, _ in entries}
        option_help = {label: help_text for label, _, help_text in entries}

        flag = AskFor.Options(
            labels,
            confirm=False,
            multiple=False,
            title=title,
            help_description=help_description,
            option_help=option_help,
        )
        if not flag.valid or flag.result is None:
            return None
        return label_to_cls[flag.result]

    @classmethod
    def _pick_pipeline_script(cls) -> ScriptRunner | None:
        from src.proj.util.cli import AskFor

        entries = cls._pipeline_script_entries()
        if not entries:
            Logger.note('No runnable pipeline scripts found.')
            return None

        labels = [label for label, _, _ in entries]
        label_to_runner = {label: runner for label, runner, _ in entries}
        option_help = {label: help_text for label, _, help_text in entries}

        flag = AskFor.Options(
            labels,
            confirm=False,
            multiple=False,
            title='Which pipeline script to run?',
            help_description=(
                'Runnable scripts from scripts/ (numbered folders, not hidden or blacklisted on this machine). '
                'The selected script opens in a new pane for parameter prompts and execution. '
                'Select « Back (q) » in the menu to return to the hub.'
            ),
            option_help=option_help,
        )
        if not flag.valid or flag.result is None:
            return None
        return label_to_runner[flag.result]

    def _dispatch_top_level(self, choice: str) -> None:
        if choice == 'Git Pull':
            from src.api.calls.source_code import GitClearPull

            Logger.note('Spawning [GitClearPull] in new pane')
            GitClearPull.spawn_in_pane(vertical=True, done_action='close')
            return

        if choice == 'Launch Streamlit App':
            from src.api.calls.app import LaunchApp

            Logger.note('Spawning [LaunchApp] in new pane')
            LaunchApp.spawn_in_pane(vertical=True, done_action='close')
            return

        if choice == 'Non-Research Operations':
            selected_cls = self._pick_direct_call(
                self._source_code_entries(),
                title='Which non-research operation to launch?',
                help_description=(
                    'Each choice spawns a DirectCall in a split pane. '
                    'Select « Back (q) » in the menu to return to the hub. '
                    'Git pull asserts on coding platforms at runtime.'
                ),
            )
            if selected_cls is not None:
                Logger.note(f'Spawning [{selected_cls.__name__}] in new pane')
                selected_cls.spawn_in_pane()
            return

        if choice == 'Research Operations':
            selected_cls = self._pick_direct_call(
                self._research_entries(),
                title='Which research operation to launch?',
                help_description=(
                    'Each choice spawns a research DirectCall in a split pane. '
                    'Select « Back (q) » in the menu to return to the hub.'
                ),
            )
            if selected_cls is not None:
                Logger.note(f'Spawning [{selected_cls.__name__}] in new pane')
                selected_cls.spawn_in_pane()
            return

        if choice == 'Run Pipeline Script':
            runner = self._pick_pipeline_script()
            if runner is None:
                return
            from src.api.calls.script import RunPipelineScript

            Logger.note(f'Spawning [RunPipelineScript] for [{runner.script_key}] in new pane')
            RunPipelineScript.spawn_in_pane(script_key=runner.script_key)
            return

        raise ValueError(f'Unknown top-level choice: {choice}')

    def run(self) -> None:
        from src.proj.util.cli import AskFor

        for loop in AskFor.LoopTillExit(ask=False, message='Launch another action?'):
            flag = AskFor.Options(
                list(_TOP_LEVEL_LABELS),
                confirm=False,
                multiple=False,
                allow_back=False,
                title='DirectCall Hub — what do you want to do?',
                help_description=(
                    'Four top-level groups. Launch Streamlit runs immediately; the other three open a submenu. '
                    'Selections spawn in a split pane while this hub keeps running. '
                    'Submenus offer « Back (q) » to return here; use /quit to exit the hub.'
                ),
                option_help=self._top_level_help(),
            )
            if not loop.set_flag(flag) or flag.result is None:
                continue

            self._dispatch_top_level(flag.result)
