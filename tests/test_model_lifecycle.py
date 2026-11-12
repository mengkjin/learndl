"""Lifecycle integration against real configuration parsing and temporary artifacts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.proj import PATH
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import ModelPath
from src.res.model.util.lifecycle import ModelLifecycle, _read, _write, _digest


class ModelLifecycleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for key, relative in [('model', 'models'), ('model_nn', 'models/nn'),
                              ('model_boost', 'models/boost'), ('model_st', 'models/st'),
                              ('lc_machine', 'local'), ('sched', 'schedule')]:
            folder = self.root / relative
            folder.mkdir(parents=True, exist_ok=True)
            self.enterContext(patch.object(PATH, key, folder))
        self.enterContext(patch.object(ModelPath, 'log_operation'))
        (PATH.sched / 'lifecycle_test.yaml').write_text(
            'model.module: gru\ninput.data.types: day\ntrain.max_epoch: 200\nmodel.window: 2400\n')

    def config(self, **kwargs):
        return ModelConfig(schedule_name='lifecycle_test', stage=1, resume=0,
                           selection=0, short_test=False, vb_level='never', **kwargs).start_model()

    def test_quick_preserves_full_target_and_resume(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        self.assertEqual(config.max_epoch, 20)
        self.assertEqual(config.window, 1200)
        self.assertEqual(_read(manager.target_file)['model']['train.max_epoch'], 200)
        self.assertEqual(_read(config.base_path.conf_file('model'))['train.max_epoch'], 200)
        resumed = ModelConfig(config.base_path, stage=2, resume=1, selection=0, vb_level='never').start_model()
        self.assertEqual(resumed.max_epoch, 20)
        self.assertEqual(resumed.window, 1200)

    def test_smaller_values_are_preserved(self):
        config = self.config(override={'train.max_epoch': 7, 'model.window': 200})
        self.assertEqual((config.max_epoch, config.window), (7, 200))

    def test_model_budget_overrides(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        manager.save(budget={'enabled': True, 'max_epoch': 12, 'window': 600, 'step': 3})
        resumed = ModelConfig(config.base_path, stage=2, resume=1, selection=0, vb_level='never').start_model()
        self.assertEqual((resumed.max_epoch, resumed.window), (12, 600))

    def test_invalid_budget_rejected(self):
        config = self.config()
        ModelLifecycle(config.base_path.base).save(budget={'enabled': True, 'step': 0})
        with self.assertRaises(ValueError):
            ModelLifecycle.prepare(config)

    def test_st_cannot_promote(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        manager.save(short_test=True, stage='production')
        with self.assertRaises(ValueError):
            ModelLifecycle.prepare(config)

    def test_grid_is_sampled_before_range_filter(self):
        config = self.config()
        config.options.start = 20170106
        config.options.end = 20170112
        self.assertEqual(ModelLifecycle.selected_nodes(config,
                         [20170103, 20170104, 20170105, 20170106, 20170109, 20170110, 20170111]),
                         [20170109, 20170111])

    def test_metadata_only_is_not_a_weight(self):
        config = self.config()
        directory = config.base_path.full_path(0, 20200102, 'best')
        directory.mkdir(parents=True)
        (directory / 'metadata.json').write_text('{}')
        self.assertFalse(config.base_path.exists(0, 20200102, 'best'))

    def artifact(self, config, date):
        directory = config.base_path.full_path(0, date, 'best')
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({'weight': torch.ones(1)}, directory / 'state_dict.pt')
        manager = ModelLifecycle(config.base_path.base)
        records = _read(manager.manifest_file)
        records[f'0:{date}:best'] = {
            'definition': _digest(ModelLifecycle.semantic(ModelLifecycle.snapshot(config))),
            'weights': manager.weights(0, date, 'best'),
        }
        _write(manager.manifest_file, records)

    def test_fill_holes_preserves_later_nodes(self):
        config = self.config()
        ModelLifecycle.configured(config)
        self.artifact(config, 20200102)
        self.artifact(config, 20220104)
        trainer = SimpleNamespace(base_path=config.base_path, config=config)
        nodes = [(20200102, 0), (20210104, 0), (20220104, 0)]
        self.assertEqual(ModelLifecycle.pending_nodes(trainer, nodes), [(20210104, 0)])
        config.model_config['train.trainer.transfer'] = True
        self.assertEqual(ModelLifecycle.pending_nodes(trainer, nodes), [(20210104, 0), (20220104, 0)])

    def test_changed_recipe_and_corruption_invalidate(self):
        config = self.config()
        self.artifact(config, 20200102)
        manager = ModelLifecycle(config.base_path.base)
        self.assertTrue(manager.compatible(config, 0, 20200102))
        config.model_config['train.max_epoch'] = 200
        self.assertFalse(manager.compatible(config, 0, 20200102))
        config.model_config['train.max_epoch'] = 20
        config.base_path.full_path(0, 20200102, 'best').joinpath('state_dict.pt').write_bytes(b'changed')
        self.assertFalse(manager.compatible(config, 0, 20200102))

    def test_production_entry_is_read_only(self):
        config = self.config()
        ModelLifecycle(config.base_path.base).save(stage='production')
        with self.assertRaises(ValueError):
            ModelLifecycle.validate_entry(config)
        config.queue_of_stages = ['test']
        ModelLifecycle.validate_entry(config)

    def test_approval_does_not_expose_exploration(self):
        config = self.config()
        with self.assertRaises(ValueError):
            ModelLifecycle.approved_path(config.base_path, registered=True)
        manager = ModelLifecycle(config.base_path.base)
        approved = config.base_path.copy().with_new_index(2)
        approved.base.mkdir()
        manager.save(approved_path=str(approved.base))
        self.assertEqual(ModelLifecycle.approved_path(config.base_path, registered=True).base, approved.base)

    def test_target_update_invalidates_acceptance(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        manager.save(accepted_request='old')
        target = _read(manager.target_file)
        target['model']['train.max_epoch'] = 300
        manager.update_target(target)
        self.assertIsNone(manager.state['accepted_request'])
        self.assertEqual(manager.state['target_version'], _digest(target))

    def test_no_partial_json_on_replace(self):
        manager = ModelLifecycle(self.root / 'atomic')
        manager.save(stage='exploration')
        with patch('src.res.model.util.lifecycle.os.replace', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                manager.save(stage='production')
        self.assertEqual(_read(manager.file)['stage'], 'exploration')
        self.assertFalse(list(manager.file.parent.glob('*.tmp')))

    def test_repair_publishes_only_after_acceptance_and_reuses_failed_candidate(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        request = {'name': config.base_path.full_name, 'base': str(manager.base),
                   'target_version': manager.state['target_version'], 'version': 'request-one',
                   'stage': 'production', 'nodes': [20200102], 'start': 20200102, 'end': 20210101}
        candidates = []

        def run(**kwargs):
            candidate_config = ModelConfig(**kwargs, vb_level='never').start_model()
            candidates.append(candidate_config.base_path.base)
            self.assertEqual(candidate_config.max_epoch, 200)
            self.assertEqual(candidate_config.window, 2400)
            ModelLifecycle.configured(candidate_config)
            self.artifact(candidate_config, 20200102)
            return SimpleNamespace(config=candidate_config,
                                   data=SimpleNamespace(model_date_list=[20200102]))

        with patch.object(ModelLifecycle, 'registrations', return_value=[request]):
            with patch.object(ModelLifecycle, '_validate_predictions', side_effect=ValueError('bad test')):
                with self.assertRaisesRegex(ValueError, 'bad test'):
                    manager.repair(request, run)
            self.assertFalse(manager.state.get('approved_path'))
            self.assertEqual(manager.state['status'], 'failed')
            with patch.object(ModelLifecycle, '_validate_predictions'):
                manager.repair(request, run)
        self.assertEqual(candidates[0], candidates[1])
        self.assertEqual(manager.state['approved_path'], str(candidates[0]))
        self.assertEqual(manager.state['stage'], 'production')
        self.assertEqual(manager.state['accepted_request'], 'request-one')
        self.assertEqual(ModelLifecycle(config.base_path.base).state['budget']['max_epoch'], 20)

    def test_repair_rejects_changed_target_without_running(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        with self.assertRaisesRegex(ValueError, 'stale'):
            manager.repair({'target_version': 'wrong'}, lambda **kwargs: self.fail('must not run'))

    def test_registered_inspection_is_idempotent_for_legacy(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        manager.save(legacy=True, stage='production')
        target = _read(manager.target_file)
        target['model']['model.beg_date'] = 20200102
        target['model']['model.end_date'] = 20200102
        _write(manager.target_file, target)
        self.artifact(config, 20200102)
        registration = {'base': str(manager.base), 'name': config.base_path.full_name, 'stage': 'production'}
        with patch.object(ModelLifecycle, 'registrations', return_value=[registration]):
            self.assertEqual(ModelLifecycle.inspect_registered(), [])
            self.assertEqual(ModelLifecycle.inspect_registered(), [])

    def test_boost_keeps_training_parameters_and_samples_nodes(self):
        (PATH.sched / 'boost_lifecycle.yaml').write_text('model.module: lgbm\nmodel.window: 240\n')
        config = ModelConfig(schedule_name='boost_lifecycle', stage=1, resume=0, selection=0,
                             short_test=False, vb_level='never').start_model()
        self.assertEqual(config.window, 240)
        config.options.start = 20200101
        config.options.end = 20201231
        self.assertEqual(ModelLifecycle.selected_nodes(config, [20200102, 20200203, 20200302]),
                         [20200102, 20200302])

    def test_frozen_split_compression_preserves_purge_boundary(self):
        config = self.config()
        split = {'train': [1, 2, 3], 'valid': [5, 6], 'test': [8]}
        self.assertEqual(ModelLifecycle.compressed_split(config, split, {3, 5, 6}),
                         {'train': [3], 'valid': [5, 6], 'test': [8]})
        with self.assertRaisesRegex(ValueError, 'no explicit train'):
            ModelLifecycle.compressed_split(config, split, {5, 6})
        ModelLifecycle(config.base_path.base).save(stage='validation')
        self.assertEqual(ModelLifecycle.compressed_split(config, split, {3, 5, 6}), split)

    def test_production_prediction_coverage_checks_each_member(self):
        import pandas as pd
        config = self.config()
        candidate = config.base_path
        folder = candidate.snapshot('pred_recorder', 'preds')
        folder.mkdir(parents=True)
        (folder / 'pred.feather').touch()
        data = SimpleNamespace(test_full_dates=[20200103, 20200106], next_model_date=lambda _: 20200106)
        trainer = SimpleNamespace(data=data, config=config)
        frame = pd.DataFrame({'model_date': [20200102, 20200102], 'model_num': [0, 0],
                              'submodel': ['best', 'best'], 'date': [20200103, 20200106], 'pred': [0.1, 0.2]})
        with patch('src.proj.Load.df', return_value=frame):
            ModelLifecycle._validate_predictions(candidate, trainer, [(20200102, 0)])
            with self.assertRaisesRegex(ValueError, 'Missing evaluated'):
                ModelLifecycle._validate_predictions(candidate, trainer, [(20200102, 0), (20200102, 1)])
        frame.loc[0, 'pred'] = float('nan')
        with patch('src.proj.Load.df', return_value=frame):
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                ModelLifecycle._validate_predictions(candidate, trainer, [(20200102, 0)])

    def test_sparse_full_repair_copies_compatible_weights(self):
        config = self.config(override={'train.max_epoch': 10, 'model.window': 600})
        manager = ModelLifecycle(config.base_path.base)
        self.artifact(config, 20200102)
        request = {'name': config.base_path.full_name, 'base': str(manager.base),
                   'target_version': manager.state['target_version'], 'version': 'sparse',
                   'stage': 'production', 'nodes': [20200102], 'start': 20200102, 'end': 20210101}

        def run(**kwargs):
            candidate = kwargs['base_path']
            self.assertTrue(candidate.full_path(0, 20200102, 'best').joinpath('state_dict.pt').is_file())
            candidate_config = ModelConfig(**kwargs, vb_level='never').start_model()
            ModelLifecycle.configured(candidate_config)
            self.assertEqual(ModelLifecycle.semantic(ModelLifecycle.snapshot(config)),
                             ModelLifecycle.semantic(candidate_config.value_dict['lifecycle_effective']))
            self.assertTrue(ModelLifecycle(candidate.base).compatible(candidate_config, 0, 20200102))
            return SimpleNamespace(config=candidate_config, data=SimpleNamespace(model_date_list=[20200102]))

        with patch.object(ModelLifecycle, 'registrations', return_value=[request]):
            with patch.object(ModelLifecycle, '_validate_predictions'):
                manager.repair(request, run)
        self.assertEqual(manager.state['status'], 'ready')

    def test_daily_plan_stops_after_publication_and_exports_alias(self):
        config = self.config(override={'model.beg_date': 20200102, 'model.end_date': 20200103})
        manager = ModelLifecycle(config.base_path.base)
        registration = {'base': str(manager.base), 'name': config.base_path.full_name, 'stage': 'production'}

        def run(**kwargs):
            candidate_config = ModelConfig(**kwargs, vb_level='never').start_model()
            ModelLifecycle.configured(candidate_config)
            self.artifact(candidate_config, 20200102)
            return SimpleNamespace(config=candidate_config, data=SimpleNamespace(model_date_list=[20200102]))

        with patch.object(ModelLifecycle, 'registrations', return_value=[registration]):
            plans = ModelLifecycle.inspect_registered()
            self.assertEqual(len(plans), 1)
            with patch.object(ModelLifecycle, '_validate_predictions'):
                manager.repair(plans[0], run)
            self.assertEqual(ModelLifecycle.inspect_registered(), [])
            approved = ModelPath(manager.state['approved_model'])
            files = approved.collect_model_archives()
            self.assertIn(manager.file, files)
            self.assertIn(approved.conf_file('production'), files)
            self.assertIn(approved.snapshot('model_manifest.json'), files)
            approved.full_path(0, 20200102, 'best').joinpath('state_dict.pt').write_bytes(b'corrupt')
            repaired = ModelLifecycle.inspect_registered()
            self.assertEqual(len(repaired), 1)
            self.assertNotEqual(repaired[0]['version'], plans[0]['version'])

    def test_sparse_and_full_modes_preserve_single_node_parameters(self):
        config = self.config(training_mode='sparse', lifecycle_options={'step': 3})
        self.assertEqual((config.max_epoch, config.window), (200, 2400))
        self.assertEqual(ModelLifecycle(config.base_path.base).state['stage'], 'exploration')
        self.assertEqual(ModelLifecycle.selected_nodes(config, [20170103, 20170104, 20170105, 20170106]),
                         [20170103, 20170106])
        full = self.config(training_mode='full')
        self.assertEqual((full.max_epoch, full.window), (200, 2400))
        self.assertEqual(ModelLifecycle.selected_nodes(full, [20170103, 20170104]), [20170103, 20170104])

    def test_mode_switch_rechecks_weights_and_archives_evaluation(self):
        import tarfile
        config = self.config(training_mode='quick')
        self.artifact(config, 20200102)
        folder = config.base_path.snapshot('pred_recorder')
        folder.mkdir(parents=True)
        (folder / 'old-prediction').write_text('old')
        resumed = ModelConfig(config.base_path, stage=1, resume=1, selection=0, short_test=False,
                              training_mode='sparse', vb_level='never').start_model()
        self.assertEqual((resumed.max_epoch, resumed.window), (200, 2400))
        manager = ModelLifecycle(config.base_path.base)
        self.assertFalse(manager.compatible(resumed, 0, 20200102))
        self.assertFalse(folder.exists())
        archives = list(config.base_path.snapshot().glob('lifecycle_previous_evaluation.*.tar'))
        self.assertEqual(len(archives), 1)
        with tarfile.open(archives[0]) as archive:
            self.assertIn('snapshot/pred_recorder/old-prediction', archive.getnames())
        self.assertTrue(config.base_path.full_path(0, 20200102, 'best').joinpath('state_dict.pt').exists())
        inherited = ModelConfig(config.base_path, stage=2, resume=1, selection=0, short_test=False,
                                vb_level='never').start_model()
        self.assertEqual((inherited.max_epoch, inherited.window), (200, 2400))
        self.assertEqual(len(list(config.base_path.snapshot().glob('lifecycle_previous_evaluation.*.tar'))), 1)

    def test_quick_explicit_caps_and_validation_override_rejection(self):
        config = self.config(lifecycle_options={'mode': 'quick', 'max_epoch': 11, 'window': 700})
        self.assertEqual((config.max_epoch, config.window), (11, 700))
        ModelLifecycle(config.base_path.base).save(stage='validation')
        config.lifecycle_options = {'mode': 'quick'}
        with self.assertRaisesRegex(ValueError, 'cannot modify'):
            ModelLifecycle.prepare(config)


if __name__ == '__main__':
    unittest.main()
