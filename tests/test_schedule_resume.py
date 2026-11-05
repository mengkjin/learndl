"""Legacy schedule recovery uses archived configuration and an explicit directory."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
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
    training_run: history.TrainingRun
    _config: ModelConfig

    def __init__(self, base_path=None, short_test: bool | None = False):
        self._config_kwargs = {'base_path': base_path, 'schedule_name': 'gru_day_new_rtn'}
        self._kwargs = dict(stage=1, resume=1, selection=0, short_test=short_test, vb_level='never')
        self.input_model_kwargs = self._kwargs

    @history.record_training
    def execute(self):
        BaseTrainer.init_config(cast(BaseTrainer, self))
        self.training_run.configured(self._config)
        return self._config


class LegacyScheduleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        source = 'model.module: gru\ninput.data.types: day\n'
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

    def auto_config(self):
        return recovery.automatic_schedule_config(
            {'schedule_name': 'gru_day_new_rtn', 'base_path': None, 'vb_level': 'never'},
            {'resume': 1, 'stage': 1, 'selection': 0, 'short_test': False,
             'resume_selection': 'latest'})

    def copy_run(self, index):
        other = self.original.with_name(self.original.name + f'@{index}')
        shutil.copytree(self.original, other)
        return other

    def test_automatic_numeric_order_deleted_and_renamed_runs(self):
        second = self.copy_run(2)
        tenth = self.copy_run(10)
        self.assertEqual(self.auto_config().base_path.base, tenth)
        shutil.rmtree(tenth)
        self.assertEqual(self.auto_config().base_path.base, second)
        renamed = second.with_name(self.original.name + '@7')
        second.rename(renamed)
        self.assertEqual(self.auto_config().base_path.base, renamed)
        shutil.rmtree(renamed)
        chosen = self.auto_config()
        self.assertEqual(chosen.base_path.base, self.original)
        chosen.start_model()
        self.assertTrue(chosen.is_resuming)

    def test_automatic_skips_mismatch_and_corruption(self):
        other = self.copy_run(2)
        file = other / 'configs' / 'algo.gru.yaml'
        algo = yaml.safe_load(file.read_text())
        algo['hidden_dim'] = [999]
        file.write_text(yaml.safe_dump(algo))
        self.assertEqual(self.auto_config().base_path.base, self.original)
        file.write_text('invalid: [')
        self.assertEqual(self.auto_config().base_path.base, self.original)
        shutil.rmtree(self.original)
        chosen = self.auto_config()
        self.assertEqual(chosen.options.resume, 0)
        self.assertFalse(chosen.base_path.base.exists())
        chosen.start_model()
        self.assertNotEqual(chosen.base_path.base, other)
        self.assertEqual(file.read_text(), 'invalid: [')

    def test_automatic_definition_changes_and_runtime_settings(self):
        original = self.source.read_text()
        for change in ('input.sequence.lens: {day: 19}', 'model.window: 19',
                       'train.trainer.learn_rate.base: 0.123',
                       'train.max_epoch: 99'):
            with self.subTest(change=change):
                self.source.write_text(original + change + '\n')
                config = self.auto_config()
                self.assertEqual(config.options.resume, 0)
                self.assertNotEqual(config.base_path.base, self.original)
        self.source.write_text(original + 'callbacks.ActivationCheckpointing: {enabled: true}\n')
        chosen = self.auto_config()
        self.assertEqual(chosen.base_path.base, self.original)
        self.assertTrue(chosen.callback_kwargs['ActivationCheckpointing']['enabled'])

    def test_automatic_trainer_initialization_pins_existing_directory(self):
        self.copy_run(2)
        trainer = LegacyTrainer()
        trainer._kwargs['resume_selection'] = 'latest'
        config = trainer.execute()
        self.assertEqual(config.base_path.model_name_index, 2)
        self.assertTrue(config.is_resuming)

    def test_automatic_rejects_inconsistent_saved_module(self):
        other = self.copy_run(2)
        saved = other / 'configs' / 'model.yaml'
        content = yaml.safe_load(saved.read_text())
        content['model.module'] = 'nn@lstm'
        saved.write_text(yaml.safe_dump(content))
        self.assertEqual(self.auto_config().base_path.base, self.original)

    def test_automatic_missing_saved_artifact_falls_back(self):
        other = self.copy_run(2)
        (other / 'archive/0/20240101/best/state_dict.pt').unlink()
        self.assertEqual(self.auto_config().base_path.base, self.original)

    def test_automatic_rechecks_configuration_before_start(self):
        from src.res.model.util import resume_validation
        validate = resume_validation.validate_resume
        saved = self.original / 'configs' / 'model.yaml'
        calls = []
        def change_once(config):
            validate(config)
            calls.append(1)
            if len(calls) == 1:
                saved.write_text(saved.read_text() + '\n# changed during validation\n')
        with patch.object(resume_validation, 'validate_resume', side_effect=change_once):
            self.assertEqual(self.auto_config().base_path.base, self.original)
        self.assertEqual(len(calls), 2)

        def change_always(config):
            validate(config)
            saved.write_text(saved.read_text() + '\n# changed again\n')
        with patch.object(resume_validation, 'validate_resume', side_effect=change_always):
            with self.assertRaisesRegex(RuntimeError, 'changed repeatedly'):
                self.auto_config()

    def test_automatic_no_directories_creates_only_after_start(self):
        shutil.rmtree(self.original)
        chosen = self.auto_config()
        self.assertEqual(chosen.options.resume, 0)
        self.assertFalse(chosen.base_path.base.exists())
        chosen.start_model()
        self.assertTrue(chosen.base_path.base.is_dir())
        self.assertFalse(chosen.is_resuming)

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

    def test_automatic_latest_uses_numeric_suffix_despite_creation_time(self):
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
            self.assertEqual(recovery.select_schedule_run('gru_day_new_rtn', short_test=False, policy='latest').base, other)

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
