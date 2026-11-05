"""Keep script discovery and ordinary startup banners free of training imports."""
from __future__ import annotations

import inspect
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / 'scripts/4_train/1_train_model.py'


class ScriptStartupTests(unittest.TestCase):
    def test_pipeline_discovery_does_not_import_training_stack(self):
        code = f'''
import importlib.abc
import runpy
import sys
from pathlib import Path
class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'sklearn', 'matplotlib'}} or fullname == 'src.api.pkgs.model':
            raise AssertionError('Premature import: ' + fullname)
sys.meta_path.insert(0, RejectTrainingImports())
for folder in Path({str(ROOT / 'scripts')!r}).iterdir():
    if folder.is_dir() and folder.name[0].isdigit():
        for script in folder.glob('*.py'):
            if not script.name.startswith('.'):
                runpy.run_path(str(script), run_name='startup_test')
'''
        result = subprocess.run(
            [sys.executable, '-c', code], cwd=ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_train_main_still_forwards_parameters(self):
        main = inspect.unwrap(runpy.run_path(str(TRAIN_SCRIPT), run_name='startup_test')['main'])
        train = Mock()
        api = SimpleNamespace(ModelAPI=SimpleNamespace(train_model=train))
        with patch.dict(sys.modules, {'src.api.pkgs.model': api}):
            main('nn/gru', True, 20200101, 20201231)
        train.assert_called_once_with('nn/gru', True, start=20200101, end=20201231)

    def test_non_training_banner_does_not_probe_device(self):
        from src.proj import MACHINE, Proj, Logger
        with patch.object(MACHINE, 'cuda_server', False), \
             patch.object(MACHINE, 'best_device', side_effect=AssertionError('GPU probe')), \
             patch.object(Logger, 'only_once') as output, \
             patch.dict(sys.modules):
            sys.modules.pop('torch', None)
            Proj.print_info()
        self.assertNotIn('Best Device', output.call_args.args[0])

    def test_cuda_server_banner_keeps_cuda_warning(self):
        from src.proj import MACHINE, Proj, Logger
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(MACHINE, 'cuda_server', True), \
             patch.object(MACHINE, 'best_device', return_value='CPU'), \
             patch.object(Logger, 'only_once') as output, \
             patch.dict(sys.modules, {'torch': torch}):
            Proj.print_info()
        self.assertEqual(output.call_args_list[0].args[0]['Best Device'], 'CPU')
        self.assertEqual(output.call_args.kwargs['printer'], 'error')


class LazyMenuTests(unittest.TestCase):
    def test_all_menus_and_dynamic_model_options_without_training(self):
        code = '''
import importlib.abc
import sys
class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sklearn', 'matplotlib', 'lightgbm', 'catboost', 'xgboost'}:
            raise AssertionError('Premature import: ' + fullname)
sys.meta_path.insert(0, RejectTrainingImports())
from src.api.calls.launcher import DirectCallHub
from src.proj.env.options import OptionsDefinition
from src.proj.util.script.param_schema import ScriptParamSchema
assert DirectCallHub._source_code_entries()
assert DirectCallHub._research_entries()
for _, runner, _ in DirectCallHub._pipeline_script_entries():
    ScriptParamSchema.from_source(runner.script)
assert OptionsDefinition.available_modules()
OptionsDefinition.available_schedules()
OptionsDefinition.available_linear_schedules()
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_defaults_match_loaded_pipeline_entries(self):
        from src.proj.util.script.param_schema import ScriptParamSchema
        # These entrypoints used to eagerly load the entire model/data/factor stack.
        for folder in ('1_autorun', '3_factor', '4_train', '5_test', '6_predict', '7_trading'):
            for path in sorted((ROOT / 'scripts' / folder).glob('*.py')):
                if path.name.startswith('.') or path.name in ('6_factor_sequence_verify.py', '2_compare_models.py'):
                    continue
                with self.subTest(script=path.name):
                    static = ScriptParamSchema.from_source(path)
                    if static is None:
                        continue
                    main = runpy.run_path(str(path), run_name='startup_test')['main']
                    runtime = ScriptParamSchema.from_script(path, main=main)
                    self.assertEqual(static.signature_defaults, runtime.signature_defaults)
                    self.assertEqual(static.params, runtime.params)

    def test_source_defaults_and_fallback(self):
        from tempfile import TemporaryDirectory
        from src.proj.util.script.param_schema import ScriptParamSchema
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'script.py'
            path.write_text("def main(a, /, b=-2, *, c=None, d=('x', 3), e): pass\n")
            schema = ScriptParamSchema.from_source(path)
            self.assertIsNotNone(schema)
            assert schema is not None
            self.assertEqual(schema.signature_defaults, {'b': -2, 'c': None, 'd': ('x', 3)})
            for source in ('def main(x=build_default()): pass\n',
                           '@custom\ndef main(x=1): pass\n'):
                path.write_text(source)
                self.assertIsNone(ScriptParamSchema.from_source(path))

    def test_cancel_does_not_load_script(self):
        from src.proj.util.cli.ask import AskFlag
        from src.proj.util.cli.script_params import run_script_interactive
        runner = SimpleNamespace(script=TRAIN_SCRIPT)
        with patch('src.api.util.backend.ScriptRunner.from_key', return_value=runner), \
             patch('src.proj.util.cli.script_params.prompt_script_kwargs', return_value=AskFlag('exit')), \
             patch('src.proj.util.filesys.dynamic_import.dynamic_modules') as load:
            run_script_interactive('4_train/1_train_model.py')
        load.assert_not_called()

    def test_confirm_loads_and_calls_script(self):
        from src.proj.util.cli.ask import AskFlag
        from src.proj.util.cli.script_params import run_script_interactive
        runner = SimpleNamespace(script=TRAIN_SCRIPT)
        main = Mock()
        flag = AskFlag('valid').set_result([{'short_test': True}])
        with patch('src.api.util.backend.ScriptRunner.from_key', return_value=runner), \
             patch('src.proj.util.cli.script_params.prompt_script_kwargs', return_value=flag), \
             patch('src.proj.util.filesys.dynamic_import.dynamic_modules', return_value=[SimpleNamespace(main=main)]) as load:
            run_script_interactive('4_train/1_train_model.py')
        load.assert_called_once_with(TRAIN_SCRIPT)
        main.assert_called_once_with(short_test=True)

    def test_schedule_discovery_rejects_duplicates(self):
        from tempfile import TemporaryDirectory
        from src.proj.env import schedules
        with TemporaryDirectory() as folder:
            roots = [Path(folder) / name for name in ('current', 'shared', 'linear')]
            for root in roots:
                root.mkdir()
            paths = SimpleNamespace(sched=roots[0], sched_shared=roots[1], sched_linear=roots[2])
            with patch.object(schedules, 'PATH', paths):
                (roots[0] / 'one.yaml').write_text('{}')
                self.assertEqual(schedules.available_schedules(), ['one'])
                self.assertEqual(schedules.find_schedule_path('one'), roots[0] / 'one.yaml')
                self.assertIsNone(schedules.find_schedule_path('missing'))
                (roots[1] / 'one.yaml').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Duplicate schedule'):
                    schedules.available_schedules()


if __name__ == '__main__':
    unittest.main()
