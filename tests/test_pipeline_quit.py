"""Pipeline panes close only after explicit Quit, retaining the rerun menu."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import Mock, patch

from src.api.calls.script import RunPipelineScript
from src.proj.util.cli.ask import AskFor, AskFlag, LoopFlag
from src.proj.util.shell.util.pausing import compose_with_done_action, EXPLICIT_QUIT_CODE

__all__ = ['PipelineQuitTest']


class PipelineQuitTest(unittest.TestCase):
    def test_rerun_same_change_parameters_then_quit(self):
        runner = SimpleNamespace(script_key='test', script=Path('/tmp/test.py'),
                                 format_path='test', header=SimpleNamespace(email=False))
        main = Mock(side_effect=[SystemExit(0), None, None])
        with (patch('src.api.calls.script.iter_runnable_scripts', return_value=[runner]),
              patch('src.api.calls.script.ScriptRunner.from_key', return_value=runner),
              patch.object(RunPipelineScript, '_load_main', return_value=main),
              patch('src.proj.util.cli.script_session.as_script_main', return_value=nullcontext()),
              patch('src.automation.scheduling.maintenance_gate.script_permit', return_value=nullcontext()),
              patch('src.proj.util.script.param_schema.ScriptParamSchema.from_source', return_value=object()),
              patch.object(AskFor, 'LoopTillExit', return_value=iter([LoopFlag(), LoopFlag(), LoopFlag()])),
              patch.object(AskFor, 'ScriptKwargs', side_effect=[
                  AskFlag('valid').set_result([{'x': 1}]), AskFlag('valid').set_result([{'x': 2}])]) as params,
              patch.object(AskFor, 'Retry', side_effect=[
                  AskFlag('valid'), AskFlag('invalid'), AskFlag('exit')]) as retry):
            with self.assertRaises(SystemExit) as error:
                RunPipelineScript(script_key='test', close_on_quit=True).run()
        self.assertEqual(error.exception.code, EXPLICIT_QUIT_CODE)
        self.assertEqual([c.kwargs for c in main.call_args_list], [{'x': 1}, {'x': 1}, {'x': 2}])
        self.assertEqual(params.call_count, 2)
        self.assertEqual(retry.call_count, 3)

    def test_only_explicit_quit_skips_shell_pause(self):
        for code in (0, 1, EXPLICIT_QUIT_CODE):
            command = compose_with_done_action(f'(exit {code})', done_action='pause_unless_quit')
            result = subprocess.run(['bash', '-c', command], input='x', text=True, capture_output=True)
            self.assertEqual('Press any key' in result.stdout, code != EXPLICIT_QUIT_CODE)
        command = compose_with_done_action('(exit 0)')
        self.assertIn('Press any key', command)

    def test_manual_quit_keeps_normal_exit_status(self):
        with patch('src.api.util.direct_call.can_exec_restart', return_value=True):
            with self.assertRaises(SystemExit) as error:
                RunPipelineScript()._handle_quit('Quit selected')
        self.assertEqual(error.exception.code, 0)
