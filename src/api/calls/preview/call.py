"""Direct call for previewing project files in the terminal."""

from __future__ import annotations

from src.api.calls.preview.reader import AutomaticReader
from src.api.calls.preview.safe_eval import SafeObjEval
from src.api.util.direct_call import DirectCall
from src.proj import Logger
from src.proj.util.cli.ask import AskFor
from src.proj.util.cli.project_path import resolve_project_path

__all__ = ['PreviewProjectFile']


class PreviewProjectFile(DirectCall):
    """Preview a project file: text snippets or loaded object inspection."""

    category = 'Files'

    @classmethod
    def get_description(cls, **kwargs) -> str:
        return (
            'Preview files under the project tree. Text files show the first 30 lines; '
            'structured files (feather, parquet, torch, tar, mmap) load via project loaders '
            'and support safe read-only obj.* expressions plus whitelisted builtins (e.g. len).'
        )

    def run(self) -> None:
        reader = AutomaticReader()
        for outer in AskFor.LoopTillExit(ask=False):
            path_flag = AskFor.ProjectPath(
                title='Preview project file',
                help_description=(
                    'Enter a project-relative path (not absolute). Tab completes paths; '
                    'type / for magic commands (e.g. /history).'
                ),
            )
            if not outer.set_flag(path_flag) or path_flag.result is None:
                continue

            try:
                resolved = resolve_project_path(path_flag.result)
            except (FileNotFoundError, ValueError) as exc:
                Logger.error(str(exc))
                continue

            try:
                result = reader.read(resolved)
            except Exception as exc:
                Logger.error(f'Failed to read {resolved}: {exc}')
                continue

            Logger.stdout(result.summary)
            if result.mode == 'text':
                continue

            self._interactive_object_loop(result.payload)

    @classmethod
    def _interactive_object_loop(cls, obj: object) -> None:
        from src.proj.util.cli.prompts import prompt_expression

        while True:
            selection = prompt_expression('obj expression')
            if selection is None or selection.lower() == 'q':
                break
            try:
                SafeObjEval.eval(selection, obj)
            except Exception as exc:
                Logger.error(str(exc))
