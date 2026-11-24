"""Lifecycle integration against real configuration parsing and temporary artifacts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

from src.proj import PATH
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import ModelPath
from src.res.model.util.lifecycle import ModelLifecycle, _read, _write, _digest, _target_digest


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
        with patch.object(ModelLifecycle, 'registrations', return_value=[{'base': str(config.base_path.base)}]), self.assertRaises(ValueError):
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

    def test_new_calendar_node_uses_identical_historical_train_and_valid_batches(self):
        from src.data.util import DataBlock
        from src.data.preprocess import PreProHistNorm
        from src.proj import CALENDAR
        from src.res.model.util.data.data_module import DataModule
        from src.res.model.util.data.module_data import ModuleData

        self.enterContext(patch.object(CALENDAR, 'update_to', return_value=20261231))
        config = self.config(training_mode='full', override={
            'model.beg_date': 20170103, 'model.end_date': 20261124, 'model.interval': 120,
            'model.window': 60, 'model.labels': ['std_lag1_10'], 'train.skip_horizon': 20,
            'input.data.prenorm.day.histnorm': True, 'env.mem_storage': True,
        })
        config.options.start = 20261124
        dates = CALENDAR.range(20260105, 20261201)
        secids = np.arange(64)
        rng = torch.Generator().manual_seed(13)
        panel = torch.rand((len(secids), len(dates), 1, 2), generator=rng) + 1

        def batches(end, *, missing_label=False):
            selected = dates <= end
            values = panel[:, torch.as_tensor(selected)].clone()
            if end > 20261124:
                values[:, torch.as_tensor(dates[selected] > 20261026)] *= -1000
            if missing_label:
                values[:, int(np.searchsorted(dates, 20261026)), :, 1] = torch.nan
            module = ModuleData.initialize(['day'], config.labels, use_data='both')
            module.blocks['day'] = DataBlock(values[..., :1], secids, dates[selected], ['close'])
            module.blocks['y'] = DataBlock(values[..., 1:], secids, dates[selected], config.labels)
            module.label_coverage = torch.ones_like(values[..., 0], dtype=torch.bool)
            module.norms['day'] = PreProHistNorm(torch.zeros(60, 1), torch.ones(60, 1))
            data = DataModule(config, 'both', vb_level='never')
            with patch.object(ModuleData, 'initialize', return_value=module), patch.object(module, 'load'):
                data.load_data()
            self.assertEqual(data.model_date_list.tolist(), [20261124])
            data.setup('fit', {'seqlens': {'day': 3}}, 20261124)
            self.assertEqual(data.y_date[-1], 20261026)
            self.assertLess(int(CALENDAR.td(int(data.y_date[-1]), 11)), 20261124)
            return {group: [loader.storage.load(key) for key in sorted(loader.keys)]
                    for group, loader in data.loader_dict.items()}

        truncated = batches(20261109)
        extended = batches(20261201)
        for group in ('train', 'valid'):
            self.assertGreater(len(truncated[group]), 0)
            for left, right in zip(truncated[group], extended[group], strict=True):
                torch.testing.assert_close(left.x, right.x)
                torch.testing.assert_close(left.y, right.y)
                torch.testing.assert_close(left.i, right.i)
        with self.assertRaisesRegex(ValueError, 'No model dates with available fit history'):
            batches(20261023)
        with self.assertRaisesRegex(ValueError, 'Missing fit labels on 20261026'):
            batches(20261109, missing_label=True)
        config.model_config['train.skip_horizon'] = 3
        with self.assertRaisesRegex(ValueError, 'crosses model date'):
            batches(20261201)

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

    def test_registered_fit_uses_daily_entry(self):
        config = self.config()
        ModelLifecycle(config.base_path.base).save(stage='production')
        self.enterContext(patch.object(ModelLifecycle, 'registrations', return_value=[{'base': str(config.base_path.base)}]))
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
        with patch.object(ModelLifecycle, 'registrations', return_value=[{'base': str(config.base_path.base)}]):
            manager.save(stage='production', legacy=True)
            self.assertEqual(ModelLifecycle.approved_path(config.base_path, registered=True).base, config.base_path.base)
            with self.assertRaisesRegex(ValueError, 'not registered'):
                ModelLifecycle.approved_path(approved, registered=True)

    def test_prediction_registry_is_exact_and_hidden_or_stale_production_cannot_opt_in(self):
        from src.proj import Const
        from src.res.model.util.core import PredictorPath
        from unittest.mock import PropertyMock
        def alias(name):
            return {'name': name, 'num': 0, 'submodel': 'best'}

        prediction = {'day': alias('gru@day'), 'average': alias('gru@average'),
                      'another_member': alias('gru@day')}
        strategy = {'prediction': prediction, 'hidden_extraction': {'hidden': alias('gru@hidden')},
                    'lifecycle': {'gru@day@2': {'stage': 'production'}, 'gru@day': {'stage': 'validation'}}}
        with patch.object(type(Const.Model), 'strategies', new_callable=PropertyMock, return_value=strategy), \
             patch.object(PredictorPath, 'MODEL_DICT', prediction):
            model = ModelPath('gru@day')
            old = ModelPath('gru@day@2')
            ModelLifecycle(model.base).save(stage='production', approved_model=old.full_name)
            ModelLifecycle(old.base).save(stage='production')
            self.assertEqual(ModelLifecycle(old.base).state['stage'], 'exploration')
            self.assertEqual([(row['name'], row['stage']) for row in ModelLifecycle.registrations()],
                             [('gru@day', 'production'), ('gru@average', 'production')])
            self.assertEqual([p.base for p in PredictorPath.SelectModels()],
                             [model.base, ModelPath('gru@average').base, model.base])
            self.assertEqual(ModelLifecycle.catalog_stage(old.full_name, 'production',
                                                         ModelLifecycle.production_names(), archived=False), 'exploration')
            with self.assertRaisesRegex(ValueError, 'not registered'):
                ModelLifecycle.approved_path(old, registered=True)

    def test_resuming_predictions_keeps_unaffected_history_and_trims_at_new_node(self):
        import pandas as pd
        from src.res.model.util.trainer.pred_recorder import PredRecorder

        recorder = PredRecorder(self.config())
        for directory, prefix in [(recorder.folder_preds, '0.'), (recorder.folder_avg_preds, '')]:
            (directory / f'{prefix}20230103.20230104.20230106.feather').write_bytes(b'later')
            (directory / f'{prefix}20200102.20200103.20200103.feather').write_bytes(b'untouched')
            pd.DataFrame({'date': [20210105, 20220104, 20220105]}).to_feather(
                directory / f'{prefix}20210104.20210105.20220105.feather')
            (directory / f'{prefix}20220104.20220105.20220106.feather').write_bytes(b'stale')

        recorder.purge_retrained_model_preds([(20220104, 0)])
        recorder.purge_retrained_model_preds([(20220104, 0)])
        for directory, prefix in [(recorder.folder_preds, '0.'), (recorder.folder_avg_preds, '')]:
            untouched = directory / f'{prefix}20200102.20200103.20200103.feather'
            trimmed = directory / f'{prefix}20210104.20210105.20220104.feather'
            later = directory / f'{prefix}20230103.20230104.20230106.feather'
            self.assertEqual(set(directory.iterdir()), {untouched, trimmed, later})
            self.assertEqual(later.read_bytes(), b'later')
            self.assertEqual(untouched.read_bytes(), b'untouched')
            self.assertEqual(pd.read_feather(trimmed)['date'].tolist(), [20210105, 20220104])

    def test_daily_repair_and_trainer_share_lock_only_on_owning_thread(self):
        from concurrent.futures import ThreadPoolExecutor
        import portalocker

        path = self.root / 'lock-test'
        def train():
            with ModelLifecycle(path).lock():
                return True

        with ThreadPoolExecutor(max_workers=1) as worker:
            with ModelLifecycle(path).lock():
                self.assertTrue(train())
                with self.assertRaises(portalocker.exceptions.LockException):
                    worker.submit(train).result()
            self.assertTrue(worker.submit(train).result())

    def test_target_update_invalidates_acceptance(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        manager.save(accepted_request='old')
        target = _read(manager.target_file)
        target['model']['train.max_epoch'] = 300
        manager.update_target(target)
        self.assertIsNone(manager.state['accepted_request'])
        from src.res.model.util.config.status_logger import rename_status_callback

        self.assertEqual(manager.state['target_version'], _target_digest(target))
        self.assertEqual(_target_digest(target), _digest(rename_status_callback(target, legacy=True)))

    def test_no_partial_json_on_replace(self):
        manager = ModelLifecycle(self.root / 'atomic')
        manager.save(stage='exploration')
        with patch('src.res.model.util.lifecycle.os.replace', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                manager.save(stage='production')
        self.assertEqual(_read(manager.file)['stage'], 'exploration')
        self.assertFalse(list(manager.file.parent.glob('*.tmp')))

    def test_daily_update_accepts_exact_directory_and_retries_after_failure(self):
        config = self.config()
        manager = ModelLifecycle(config.base_path.base)
        request = {'name': config.base_path.full_name, 'base': str(manager.base),
                   'target_version': manager.state['target_version'], 'version': 'request-one',
                   'stage': 'production', 'nodes': [20200102], 'start': 20200102, 'end': 20210101}
        candidates = []

        def run(**kwargs):
            candidate_config = ModelConfig(**kwargs, vb_level='never').start_model()
            candidates.append(candidate_config.base_path.base)
            self.assertEqual(kwargs['use_data'], 'both')
            self.assertEqual(candidate_config.base_path.base, manager.base)
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
            with self.assertRaisesRegex(ValueError, 'not passed acceptance'):
                ModelLifecycle.approved_path(config.base_path, registered=True)
            self.assertFalse(manager.state.get('approved_path'))
            self.assertEqual(manager.state['status'], 'failed')
            with patch.object(ModelLifecycle, '_validate_predictions'):
                manager.repair(request, run)
        self.assertEqual(candidates[0], candidates[1])
        self.assertEqual(manager.state['approved_path'], str(candidates[0]))
        with patch.object(ModelLifecycle, 'registrations', return_value=[request]):
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

    def test_november_daily_update_resumes_only_due_node_with_stale_fit_dump(self):
        from src.proj import CALENDAR
        from src.res.model.util.data.module_data import ModuleData

        model_day = 20261124
        previous = int(CALENDAR.range(20170103, model_day, until_today=False)[::120][-2])
        config = self.config(training_mode='full', override={'model.beg_date': 20170103, 'model.interval': 120})
        model = config.base_path
        manager = ModelLifecycle(model.base)
        registration = {'base': str(manager.base), 'name': model.full_name, 'stage': 'production', 'start': previous}
        self.enterContext(patch.object(ModelLifecycle, 'registrations', return_value=[registration]))
        self.enterContext(patch.object(CALENDAR, 'updated', return_value=20261109))
        self.enterContext(patch.object(ModuleData, 'fit_end_date', side_effect=AssertionError('fit-only gate')))
        manager.save(stage='production', legacy=True)
        self.artifact(config, previous)
        with patch.object(CALENDAR, 'update_to', return_value=20261123):
            self.assertEqual(ModelLifecycle.inspect_registered(), [])
        with patch.object(CALENDAR, 'update_to', return_value=model_day):
            request, = ModelLifecycle.inspect_registered()
            self.assertEqual(request['nodes'], [previous, model_day])

            def run(**kwargs):
                self.assertEqual(kwargs['use_data'], 'both')
                resumed = ModelConfig(**kwargs, vb_level='never').start_model()
                ModelLifecycle.configured(resumed)
                trainer = SimpleNamespace(base_path=model, config=resumed,
                                          data=SimpleNamespace(model_date_list=request['nodes']))
                self.assertEqual(ModelLifecycle.pending_nodes(trainer, [(previous, 0), (model_day, 0)]),
                                 [(model_day, 0)])
                self.artifact(resumed, model_day)
                return trainer

            with patch.object(ModelLifecycle, '_validate_predictions'):
                manager.repair(request, run)
            self.assertEqual(ModelLifecycle.inspect_registered(), [])
            self.assertEqual(manager.state['approved_model'], model.full_name)

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

    def test_sparse_full_update_preserves_compatible_weights(self):
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

    def test_daily_plan_stops_after_acceptance_and_exports_registered_model(self):
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

    def test_incomplete_request_is_rejected_and_old_incomplete_approval_is_repaired(self):
        config = self.config(override={'model.beg_date': 20200102, 'model.end_date': 20200103, 'model.interval': 1})
        manager = ModelLifecycle(config.base_path.base)
        registration = {'base': str(manager.base), 'name': config.base_path.full_name, 'stage': 'production'}
        self.enterContext(patch.object(ModelLifecycle, 'registrations', return_value=[registration]))
        cutoff = ModelLifecycle.fit_sample_cutoff(config, 20200103)
        with patch('src.proj.CALENDAR.updated', return_value=cutoff - 1):
            self.assertEqual(ModelLifecycle.inspect_registered(), [])
        with patch('src.proj.CALENDAR.updated', return_value=cutoff):
            request, = ModelLifecycle.inspect_registered()
        self.assertEqual(request['nodes'], [20200102, 20200103])

        def run(**kwargs):
            candidate = ModelConfig(**kwargs, vb_level='never').start_model()
            self.artifact(candidate, 20200102)
            return SimpleNamespace(config=candidate, data=SimpleNamespace(model_date_list=[20200102]))

        with self.assertRaisesRegex(ValueError, 'omitted requested model dates.*20200103'):
            manager.repair(request, run)
        self.assertFalse(manager.state.get('approved_model'))
        # Emulate a pre-fix publication with an internally consistent but incomplete manifest.
        active = config.base_path
        records = _read(ModelLifecycle(active.base).manifest_file)
        _write(active.snapshot('evaluation_manifest.json'), {'request': request, 'artifacts': _digest(records)})
        manager.save(legacy=False, stage='production', approved_model=active.full_name,
                     accepted_request=request['version'])
        repaired, = ModelLifecycle.inspect_registered()
        self.assertEqual(repaired['nodes'], request['nodes'])
        self.assertNotEqual(repaired['version'], request['version'])

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
