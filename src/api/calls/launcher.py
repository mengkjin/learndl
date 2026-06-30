"""DirectCall launcher hub."""

from __future__ import annotations

from typing import Type

from src.api.util.direct_call import DirectCall
from src.proj import Logger

__all__ = ['DirectCallHub']


class DirectCallHub(DirectCall):
    """Interactive hub: pick a DirectCall and spawn it in a new pane."""

    category = 'Basic'

    @classmethod
    def _entries(cls) -> list[tuple[str, Type[DirectCall]]]:
        from src.api.calls.app import LaunchApp
        from src.api.calls.data import ReconstructPreprocessedData
        from src.api.calls.files import ModelArchiveOperations, ProjectAutoFix
        from src.api.calls.script import RunPipelineScript
        from src.api.calls.source_code import CheckCodeIssues , GitClearPull
        from src.api.calls.test import TestCode
        from src.api.calls.dashboard import Tensorboard, OptunaDashboard

        return [
            ('Launch Streamlit App', LaunchApp),
            ('Clear Changes and Git Pull', GitClearPull),
            ('Test Code', TestCode),
            ('Check Code Issues', CheckCodeIssues),
            ('Project AutoFix', ProjectAutoFix),
            ('Reconstruct Preprocessed Data', ReconstructPreprocessedData),
            ('Model Archive Operations', ModelArchiveOperations),
            ('Tensorboard', Tensorboard),
            ('Optuna Dashboard', OptunaDashboard),
            ('Run Pipeline Script', RunPipelineScript),
        ]

    def run(self) -> None:
        from src.proj.util.cli import AskFor

        entries = self._entries()
        labels = [label for label, _ in entries]
        label_to_cls = {label: direct_call_cls for label, direct_call_cls in entries}
        option_help = {
            'Launch Streamlit App': 'Start the interactive pipeline UI in a new pane.',
            'Clear Changes and Git Pull': 'Discard local git changes and pull latest main.',
            'Test Code': 'Run project smoke tests (logger, quick train, factors).',
            'Check Code Issues': 'Static code hygiene and ruff/pyright checks.',
            'Project AutoFix': 'Config fixes, WezTerm template sync, log cleanup.',
            'Reconstruct Preprocessed Data': 'Rebuild fit/predict data frames for a PrePros key.',
            'Model Archive Operations': 'Archive, resume, rename, or inspect model folders.',
            'Tensorboard': 'Browse training logs via TensorBoard.',
            'Optuna Dashboard': 'Browse hyperparameter study SQLite databases.',
            'Run Pipeline Script': 'Pick and run a numbered script from scripts/.',
        }

        for loop in AskFor.LoopTillExit(ask=False , message='Launch another DirectCall?'):
            flag = AskFor.Options(
                labels,
                confirm=False,
                multiple=False,
                title='Which DirectCall to launch in a new pane?',
                help_description='Each choice spawns the selected DirectCall in a split pane; this hub keeps running.',
                option_help=option_help,
            )
            if not loop.set_flag(flag) or flag.result is None:
                continue

            selected_cls = label_to_cls[flag.result]
            Logger.note(f'Spawning [{selected_cls.__name__}] in new pane')
            selected_cls.spawn_in_pane()
