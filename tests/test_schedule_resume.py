"""Legacy schedule recovery uses archived configuration and an explicit directory."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from src.proj import PATH
from src.res.model.util import schedule_resume as recovery
from src.res.model.util import training_history as history
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import ModelPath
from src.res.model.util.trainer.base_trainer import BaseTrainer


class LegacyTrainer:
    def __init__(self, base_path=None, short_test=False):
        self._config_kwargs = {'base_path': base_path, 'schedule_name': 'gru_day_new_rtn'}
        self._kwargs = dict(stage=1, resume=1, selection=0, short_test=short_test, vb_level='never')
        self.input_model_kwargs = self._kwargs

    @history.record_training
    def execute(self):
        BaseTrainer.init_config(self)
        self.training_run.configured(self._config)
        return self._config


class LegacyScheduleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        source = (PATH.sched / 'gru_day_new_rtn.yaml').read_text()
        for key, relative in [('model', 'models'), ('model_nn', 'models/nn'),
                              ('model_boost', 'models/boost'), ('model_st', 'models/st'),
                              ('lc_machine', 'local'), ('sched', 'schedule')]:
            folder = self.root / relative
            folder.mkdir(parents=True, exist_ok=True)
            self.enterContext(patch.object(PATH, key, folder))
        self.source = PATH.sched / 'gru_day_new_rtn.yaml'
        self.source.write_text(source)
        self.enterContext(patch.object(ModelPath, 'log_operation'))
        self.enterContext(patch.object(history, 'git', return_value=None))
        self.enterContext(patch.object(history.Save, 'async_wait_all'))
        config = ModelConfig(schedule_name='gru_day_new_rtn', stage=1, resume=0, selection=0,
                             short_test=False, vb_level='never').start_model()
        self.original = config.base_path.base
        self.expected_algo = dict(config.algo_config.Param)
        self.expected_model = dict(config.model_config.Param)
        artifact = self.original / 'archive/0/20240101/best/state_dict.pt'
        artifact.parent.mkdir(parents=True)
        torch.save({'weight': torch.ones(1)}, artifact)

    def test_numbered_directory_and_changed_current_module(self):
        target = self.original.with_name(self.original.name + '@3')
        self.original.rename(target)
        self.source.write_text('model:\n  module: lgbm\n')
        with history.collect_training_runs() as runs:
            config = LegacyTrainer().execute()
        self.assertEqual(config.base_path.base, target)
        self.assertEqual(dict(config.algo_config.Param), self.expected_algo)
        self.assertEqual(dict(config.model_config.Param), self.expected_model)
        self.assertEqual(runs[0]['model_path'], str(target))
        self.assertEqual(runs[0]['status'], 'success')
        self.assertIn('lgbm', runs[0]['schedule_source']['content'])

    def test_multiple_candidates_require_explicit_choice(self):
        other = self.original.with_name(self.original.name + '@2')
        shutil.copytree(self.original, other)
        with patch.object(recovery.sys.stdin, 'isatty', return_value=True), \
             patch('src.proj.util.cli.AskFor.Options', return_value=SimpleNamespace(valid=True, result=str(other))) as prompt:
            config = LegacyTrainer().execute()
        self.assertEqual(config.base_path.base, other)
        self.assertEqual(set(prompt.call_args.args[0]), {str(self.original), str(other)})
        with patch.object(recovery.sys.stdin, 'isatty', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Multiple training directories'):
                LegacyTrainer().execute()

    def test_automatic_latest_uses_creation_log_then_saved_config_time(self):
        other = self.original.with_name(self.original.name + '@2')
        shutil.copytree(self.original, other)
        models = [ModelPath(self.original), ModelPath(other)]
        for index, model in enumerate(models):
            os.utime(model.conf_file('model'), (100 + index, 100 + index))
        with patch.object(ModelPath, 'log_file', new_callable=lambda: property(lambda _: SimpleNamespace(read=lambda: []))):
            self.assertEqual(recovery.select_schedule_run('gru_day_new_rtn', short_test=False, policy='latest').base, other)
        from datetime import datetime
        def entries(model):
            timestamp = datetime.fromtimestamp(200 if model.base == self.original else 100)
            return SimpleNamespace(read=lambda: [SimpleNamespace(timestamp=timestamp, title='create_model_path')])
        with patch.object(ModelPath, 'log_file', property(entries)):
            self.assertEqual(recovery.select_schedule_run('gru_day_new_rtn', short_test=False, policy='latest').base, self.original)

    def test_cancel_does_not_start_or_modify_models(self):
        other = self.original.with_name(self.original.name + '@2')
        shutil.copytree(self.original, other)
        with patch.object(recovery.sys.stdin, 'isatty', return_value=True), \
             patch('src.proj.util.cli.AskFor.Options', return_value=SimpleNamespace(valid=False, result=None)), \
             patch.object(ModelConfig, 'start_model') as start:
            with self.assertRaises(InterruptedError):
                LegacyTrainer().execute()
            start.assert_not_called()

    def test_recorded_path_is_used_without_selection(self):
        with patch.object(recovery, 'select_schedule_run', side_effect=AssertionError('must not choose')):
            config = LegacyTrainer(ModelPath(self.original), short_test=None).execute()
        self.assertEqual(config.base_path.base, self.original)

    def test_missing_directory_does_not_start_new_training(self):
        shutil.rmtree(self.original)
        with self.assertRaisesRegex(FileNotFoundError, 'no existing training directory'):
            LegacyTrainer().execute()
        self.assertEqual(list(PATH.model_nn.iterdir()), [])

    def test_bad_saved_configs_fail_and_preserve_selected_path(self):
        for name in ('model', 'schedule', 'algo.gru'):
            with self.subTest(name=name):
                file = self.original / 'configs' / f'{name}.yaml'
                content = file.read_bytes()
                file.unlink()
                observed = []
                with history.collect_training_runs(on_update=lambda data: observed.append(dict(data))) as runs:
                    with self.assertRaisesRegex(FileNotFoundError, 'saved configuration is missing'):
                        LegacyTrainer().execute()
                self.assertEqual(runs[0]['model_path'], str(self.original))
                self.assertEqual(runs[0]['status'], 'failed')
                self.assertTrue(any(row['status'] == 'running' and row['model_path'] == str(self.original)
                                    for row in observed))
                file.write_bytes(content)
        (self.original / 'configs/schedule.yaml').write_text('[invalid')
        with self.assertRaisesRegex(ValueError, 'corrupt'):
            LegacyTrainer().execute()

    def test_short_test_candidates_are_separate(self):
        short = PATH.model_st / f'nn@{self.original.name}'
        shutil.copytree(self.original, short)
        self.assertEqual([p.base for p in recovery.find_schedule_runs('gru_day_new_rtn', short_test=False)],
                         [self.original])
        self.assertEqual([p.base for p in recovery.find_schedule_runs('gru_day_new_rtn', short_test=True)], [short])
        self.source.write_text('model:\n  module: lgbm\n')
        saved_file = short / 'configs/model.yaml'
        saved = yaml.safe_load(saved_file.read_text())
        saved['env.random_seed'] = 91827
        saved_file.write_text(yaml.safe_dump(saved))
        config = LegacyTrainer(short_test=True).execute()
        self.assertEqual(config.model_config.Param['env.random_seed'], 91827)
        self.assertEqual(config.base_path.base, short)
        self.assertEqual(config.model_module, 'gru')

    def test_new_training_does_not_resolve_legacy_directory(self):
        trainer = LegacyTrainer()
        trainer._kwargs['resume'] = 0
        with patch.object(recovery, 'select_schedule_run', side_effect=AssertionError('not resuming')):
            config = trainer.execute()
        self.assertNotEqual(config.base_path.base, self.original)
        self.assertTrue(self.original.is_dir())

    def test_failure_before_selection_is_still_in_general_history(self):
        shutil.rmtree(self.original)
        with self.assertRaises(FileNotFoundError):
            LegacyTrainer().execute()
        saved = json.loads(next((PATH.lc_machine / 'training_history').glob('*/run.json')).read_text())
        self.assertEqual(saved['status'], 'failed')
        self.assertIsNone(saved['model_path'])


if __name__ == '__main__':
    unittest.main()
