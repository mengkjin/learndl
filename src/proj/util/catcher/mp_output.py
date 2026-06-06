"""Stdout/stderr capture to memory, logs, HTML, markdown, and warning interception."""
from __future__ import annotations
import json , os , re , shutil
import multiprocessing as mp

from pathlib import Path
from typing import Any

from src.proj.env import PATH

from .basic import OutputCatcher , TimedOutput , DeflectorGroup

__all__ = ['MPOutputCatcher']

class MPOutputCatcher(OutputCatcher):
    """
    Process-pool worker stdout/stderr capture; exports TimedOutput records to disk per task.

    Does not touch ``HtmlCatcher.PrimaryInstance``. Started once per worker via pool initializer.
    """

    PrimaryInstance : MPOutputCatcher | None = None
    RunID: str | None = None
    MpID: str | None = None

    export_root = PATH.logs.joinpath('catcher' , 'mp_output')
    keep_original = True

    def __init__(self , run_id: str , mp_id: str):
        self.run_id = run_id
        self.mp_id = mp_id
        self.export_dir = self.export_root.joinpath(run_id , mp_id)
        self.export_dir.mkdir(parents=True , exist_ok=True)
        self.outputs: list[TimedOutput] = []
        self.deflectors: DeflectorGroup | None = None

    def __enter__(self) -> MPOutputCatcher:
        self.deflectors = DeflectorGroup(self , self.keep_original).start_catching()
        return self

    def __exit__(self , exc_type: Any , exc_val: Any , exc_tb: Any) -> None:
        if self.deflectors is not None:
            self.deflectors.end_catching()
            self.deflectors = None

    def add_output(self , content: str | Any , output_type: str | None = None) -> None:
        if output_type is None and not isinstance(content , str):
            self.logger.warning(f'MPOutputCatcher skips non-text output: {type(content)}')
            content = str(content)
        output = TimedOutput.create(content , output_type)
        if not output or (self.outputs and output.equivalent(self.outputs[-1])):
            return
        self.outputs.append(output)

    def write_stdout(self , text: str) -> None:
        if text := text.strip('\n'):
            self.add_output(text , 'stdout')

    def write_stderr(self , text: str) -> None:
        if text := text.strip('\n'):
            self.add_output(text , 'stderr')

    def get_contents(self) -> list[dict[str, Any]]:
        return [o.to_record() for o in self.outputs]

    def export_task(self , task_key: str) -> Path | None:
        """Flush in-memory outputs to ``{export_dir}/{task_key}.jsonl`` and clear buffer."""
        if not self.outputs:
            return None
        path = self.export_dir.joinpath(f'{self._safe_mp_task_filename(task_key)}.jsonl')
        with path.open('w' , encoding='utf-8') as f:
            for output in self.outputs:
                f.write(json.dumps(output.to_record() , ensure_ascii=False) + '\n')
        self.outputs.clear()
        return path

    def flush_stdout(self) -> None:
        ...

    def flush_stderr(self) -> None:
        ...

    @staticmethod
    def _safe_mp_task_filename(task_key: str) -> str:
        name = re.sub(r'[^\w\-.]+' , '_' , str(task_key)).strip('_')
        return (name[:200] if name else 'task')

    @classmethod
    def pool_initializer(cls , run_id: str | None = None) -> None:
        """``ProcessPoolExecutor`` worker: attach :class:`MPOutputCatcher` for this run."""
        if run_id is None or mp.current_process().name == 'MainProcess':
            return
        MPOutputCatcher.export_root.joinpath(run_id).mkdir(parents=True , exist_ok=True)
        proc = mp.current_process()
        ident = proc._identity[0] if proc._identity else 0
        cls.RunID = run_id
        cls.MpID = f'w{ident}_{os.getpid()}'
        cls.PrimaryInstance = MPOutputCatcher(run_id , cls.MpID)
        cls.PrimaryInstance.__enter__()

    @classmethod
    def merge_into_html(cls , run_id: str | None = None , *, keep_on_error: bool = False, indent: int = 0) -> None:
        """
        Load worker ``jsonl`` logs for ``run_id`` and append to ``HtmlCatcher.PrimaryInstance``.

        Parent-process ``Logger.stdout`` banners bracket the worker block (plan B).
        """
        from src.proj.util.catcher import HtmlCatcher
        if HtmlCatcher.PrimaryInstance is None or run_id is None:
            return

        run_dir = MPOutputCatcher.export_root.joinpath(run_id)
        if not run_dir.is_dir():
            return

        mp_ids = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
        worker_outputs: list[TimedOutput] = []
        n_files = 0
        for mp_id in mp_ids:
            mp_prefix = ' ' * (indent + 1) * 2 + '--> ' + f'[{run_id}]: '
            for path in sorted((run_dir / mp_id).glob('*.jsonl')):
                n_files += 1
                try:
                    for line in path.read_text(encoding='utf-8').splitlines():
                        if line.strip():
                            worker_outputs.append(TimedOutput.from_record(json.loads(line) , prefix=mp_prefix))
                except Exception as e:
                    cls.logger.warning(f'Failed to load worker log {path}: {e}' , indent=indent)

        if not worker_outputs:
            cls.logger.alert1(f'No multiprocessing worker output under {run_dir}' , indent=indent)
            if not keep_on_error:
                shutil.rmtree(run_dir , ignore_errors=True)
            return

        worker_outputs.sort(key=lambda o: o.sort_key)
        cls.logger.note(f'Multiprocessing worker output begin (run_id={run_id}, workers={mp_ids}, chunk_files={n_files})' , indent=indent)
        HtmlCatcher.PrimaryInstance.outputs.extend(worker_outputs)
        cls.logger.note(f'Multiprocessing worker output end (run_id={run_id})' , indent=indent)
        if not keep_on_error:
            shutil.rmtree(run_dir , ignore_errors=True)

    @classmethod
    def export_current_task(cls , task_key: str) -> Path | None:
        """Export worker catcher buffer after one pool task (no-op in main process)."""
        if cls.PrimaryInstance is None:
            return None
        return cls.PrimaryInstance.export_task(task_key)