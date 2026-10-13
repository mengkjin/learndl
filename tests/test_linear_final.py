"""Linear schedules stay non-null while supporting checkpoint-free explicit resume."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml

from src.proj import DB, PATH
from src.res.model.util.config import ModelConfig
from src.res.model.util.config.config import ScheduleConfig
from src.res.model.util.core import ModelPath
from src.res.model.util.final_source import FinalSource, FinalInputLoader, final_weights, prepare_final_values
from src.res.model.model_module.module.module_selector import get_predictor_module
from src.res.model.model_module.application.predictor import ArchivedPredictorModel
from src.res.model.util.trainer.base_trainer import BaseTrainer


class FinalValuesTest(unittest.TestCase):
    def test_pred_duplicate_rows_and_sparse_dates(self):
        raw = pl.DataFrame({'secid': [10, 10, 20], 'date': [20240102, 20240102, 20240104],
                            'score': [2., 2., 6.]})
        config = SimpleNamespace(input_final_types=['pred@published'], input_final_weights='equal',
                                 input_final_precursor='none', input_final_refresh=False)
        dates = [20240102, 20240103, 20240104]
        with patch.object(DB, 'loads_pl', return_value=raw) as load, patch.object(FinalSource, 'validate'):
            loader = FinalInputLoader(config)
            block = loader.blocks(dates, [10, 20], dates)['pred@published']
            expected = torch.tensor([[2., float('nan'), float('nan')],
                                     [float('nan'), float('nan'), 6.]], dtype=block.values.dtype)
            torch.testing.assert_close(block.values[:, :, 0, 0], expected, equal_nan=True)
            self.assertEqual(loader.coverage_by_date, {20240102: 1, 20240103: 0, 20240104: 1})
            self.assertEqual(load.call_count, 1)
            np.testing.assert_array_equal(load.call_args.kwargs['dates'], dates)
            load.return_value = raw.with_columns(pl.Series('score', [2., 3., 6.]))
            with self.assertRaisesRegex(ValueError, r'pred@published 20240102..20240104:.*conflicting.*20240102'):
                FinalSource.parse('pred@published').frame(dates)
            load.return_value = pl.DataFrame()
            self.assertTrue(FinalSource.parse('pred@published').frame(dates).empty)

    def test_raw_sum_mean_difference_and_zero_weight(self):
        raw = torch.tensor([[[2., 4.]], [[6., 8.]]])
        for weights, expected in [([1, 1], [6., 14.]), ('equal', [3., 7.]), ([1, -1], [-2., -2.])]:
            actual = prepare_final_values(raw, 'none', weights).sum(-1).flatten()
            torch.testing.assert_close(actual, torch.tensor(expected))
        raw[0, 0, 1] = torch.nan
        result = prepare_final_values(raw, 'none', [1, 0])
        self.assertEqual(result[0, 0].tolist(), [2., 0.])
        self.assertTrue(prepare_final_values(raw, 'none', 'equal')[0].isnan().all())

    def test_normal_clips_and_handles_constant_and_missing(self):
        raw = torch.zeros(100, 2, 2)
        raw[-1, 0, 0] = 1000
        raw[0, 1, 0] = torch.inf
        value = prepare_final_values(raw, 'normal', [1, 1])
        self.assertEqual(value[-1, 0, 0].item(), 3.5)
        self.assertTrue((value[:, 0, 1] == 0).all())
        self.assertTrue(value[0, 1].isnan().all())
        self.assertTrue((value[1:, 1] == 0).all())
        # No second standardization after clipping.
        self.assertNotAlmostEqual(value[:, 0, 0].mean().item(), 0.)

    def test_percentile_ties_and_weight_errors(self):
        raw = torch.tensor([1., 1., 3.]).reshape(3, 1, 1)
        result = prepare_final_values(raw, 'percentile', 'equal')
        torch.testing.assert_close(result.flatten(), torch.tensor([.5, .5, 1.]))
        for weights in ([1], [[1, 2]], [1, float('nan')], 'sum', [True, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                final_weights(weights, 2)

    def test_model_source_rolling_dates_columns_and_refresh(self):
        source = Mock(model_dates=np.array([20240102, 20240104]))
        source.pred_values.side_effect = lambda dates, model_date, **kw: pl.DataFrame({
            'date': list(dates), 'secid': [10] * len(dates),
            'pred.0': [float(model_date)] * len(dates), 'pred.1': [2.] * len(dates)})
        ref = FinalSource.parse('model_pred@gru@demo@1@0@best@pred.0')
        with patch.object(ArchivedPredictorModel, 'from_model_reference', return_value=source):
            result = ref.frame(np.array([20240102, 20240103, 20240104, 20240105]), refresh=True)
            self.assertEqual(result.date.tolist(), [20240103, 20240104, 20240105])
            self.assertEqual(result.value.tolist(), [20240102., 20240102., 20240104.])
            self.assertTrue(all(not c.kwargs['load_first'] for c in source.pred_values.call_args_list))
            with self.assertRaisesRegex(ValueError, 'multiple output'):
                FinalSource.parse('model_pred@gru@demo').frame([20240103])
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                FinalSource.parse('model_pred@gru@demo@pred.5').frame([20240103])

    def test_refresh_of_multiple_heads_only_refreshes_raw_model_once(self):
        config = SimpleNamespace(input_final_types=['model_pred@gru@demo@pred.0', 'model_pred@gru@demo@pred.1'],
                                 input_final_weights='equal', input_final_precursor='none', input_final_refresh=True)
        loader = FinalInputLoader(config)
        frame = pd.DataFrame({'secid': [10], 'date': [20240103], 'value': [1.]})
        with patch.object(FinalSource, 'frame', return_value=frame) as read:
            loader.blocks([20240103], [10], [20240103])
            self.assertEqual([call.kwargs['refresh'] for call in read.call_args_list], [True, False])
            loader.blocks([20240103], [10], [20240103])
            self.assertEqual(read.call_count, 2)

    def test_pred_factor_loaders_and_zero_weight_missing_source(self):
        from src.data.util import DataBlock
        from src.res.factor.calculator import FactorCalculator
        block = DataBlock.from_pandas(pd.DataFrame({'secid': [20, 10], 'date': [20240102] * 2, 'score': [2., 1.]}))
        with patch.object(DB, 'loads_pl', return_value=pl.from_pandas(block.to_dataframe().reset_index())) as load, \
             patch.object(FinalSource, 'validate'), \
             patch.object(ArchivedPredictorModel, 'from_model_reference', side_effect=AssertionError('No inference')):
            result = FinalSource.parse('pred@published@2').frame([20240102])
            self.assertEqual(result.value.tolist(), [1., 2.])
            np.testing.assert_array_equal(load.call_args.kwargs['dates'], [20240102])
        with patch.object(FactorCalculator, 'all_factors', return_value=['known']), \
             patch.object(FinalSource, 'validate'), \
             patch('src.res.factor.calculator.StockFactorHierarchy.get_factor') as lookup:
            lookup.return_value.Loads.return_value = block.to_dataframe().reset_index()
            result = FinalSource.parse('factor@known').frame([20240102])
            self.assertEqual(result.value.tolist(), [1., 2.])
            self.assertFalse(lookup.return_value.Loads.call_args.kwargs['normalize'])
            with self.assertRaisesRegex(ValueError, 'Unknown factor'):
                FinalSource.parse('factor@unknown').frame([20240102])
        config = SimpleNamespace(input_final_types=['pred@a', 'pred@missing'], input_final_weights=[1, 0],
                                 input_final_precursor='none', input_final_refresh=False)
        with patch.object(FinalSource, 'frame', return_value=result) as read:
            blocks = FinalInputLoader(config).blocks([20240102], [10, 20], [20240102])
            self.assertEqual(read.call_count, 1)
            self.assertEqual(blocks['pred@missing'].values.flatten().tolist(), [0., 0.])

    def test_loader_aligns_sources_reuses_raw_cache_and_preserves_full_day(self):
        config = SimpleNamespace(input_final_types=['pred@a', 'pred@b'], input_final_weights=[1, 1],
                                 input_final_precursor='normal', input_final_refresh=False)
        def frame(source, dates, **kwargs):
            offset = 10 if source.reference == 'b' else 0
            return pd.DataFrame([(secid, date, value + offset) for date in dates
                                 for secid, value in [(30, 3.), (10, 1.), (20, 2.)]],
                                columns=['secid', 'date', 'value'])
        loader = FinalInputLoader(config)
        with patch.object(FinalSource, 'frame', autospec=True, side_effect=frame) as read:
            blocks = loader.blocks([20240102], [30, 10, 20], [20240102, 20240103])
            expected = torch.tensor([1.2247449, -1.2247449, 0.], dtype=blocks['pred@a'].values.dtype)
            torch.testing.assert_close(blocks['pred@a'].values[:, 0, 0, 0], expected)
            self.assertTrue(blocks['pred@a'].values[:, 1].isnan().all())
            loader.blocks([20240102], [30, 10, 20], [20240102])
            self.assertEqual(read.call_count, 2)


class LinearScheduleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for key, relative in [('model', 'models'), ('model_nn', 'models/nn'), ('model_boost', 'models/boost'),
                              ('model_st', 'models/st'), ('model_linear', 'models/linear'),
                              ('sched_linear', 'schedules/linear')]:
            folder = self.root / relative
            folder.mkdir(parents=True)
            self.enterContext(patch.object(PATH, key, folder))
        self.enterContext(patch.object(ModelPath, 'log_operation'))
        self.source_validation = self.enterContext(patch.object(FinalSource, 'validate'))
        self.enterContext(patch('builtins.input', side_effect=AssertionError('No prompts')))
        self.schedule = PATH.sched_linear / 'linear_unit.yaml'
        self.definition = {'model': {'module': 'linear'}, 'input': {'type': 'final',
            'final.types': ['pred@a', 'pred@b'], 'final.precursor': 'none', 'final.weights': [1, 1]}}
        self.save_schedule()

    def save_schedule(self):
        self.schedule.write_text(yaml.safe_dump(self.definition))

    def config(self, resume=-1, **kwargs):
        return ModelConfig(schedule_name='linear_unit', short_test=False, resume=resume,
                           vb_level='never', **kwargs).start_model()

    def test_fullname_non_null_stage_path_and_single_model(self):
        cfg = self.config()
        self.assertEqual(cfg.base_path.full_name, 'linear@linear_unit')
        self.assertEqual(cfg.base_path.base, PATH.model_linear / 'linear_unit')
        self.assertFalse(cfg.is_null_model)
        self.assertFalse(cfg.requires_fit)
        self.assertFalse(cfg.is_resuming)
        self.assertFalse(cfg.base_path.is_resumable)
        self.assertEqual(cfg.queue_of_stages, ['data', 'test'])
        self.assertEqual(cfg.model_num_list, [0])
        self.assertEqual(cfg.submodels, ['best'])
        self.assertEqual(cfg.input_final_weights, [1, 1])
        self.assertEqual(ModelPath(cfg.base_path.base).full_name, cfg.base_path.full_name)
        self.assertEqual(ModelPath('st@linear@linear_unit').base, PATH.model_st / 'linear@linear_unit')
        for stage in (0, 1, 2):
            self.assertEqual(self.config(stage=stage).queue_of_stages, ['data', 'test'])
        for bad in ('linear@linear_unit@2', 'linear@linear@linear_unit'):
            with self.assertRaises(ValueError):
                ModelPath(bad)

    def test_path_entry_defaults_and_factor_remain_independent(self):
        cfg = self.config()
        by_path = ModelConfig(base_path=cfg.base_path, short_test=False, resume=1, vb_level='never').start_model()
        self.assertEqual(by_path.base_path.full_name, cfg.base_path.full_name)
        self.assertEqual(by_path.full_module_name, 'linear@linear')
        self.assertTrue(by_path.is_resuming)
        factor = ModelPath('factor@unit_factor')
        self.assertTrue(factor.is_null_model)
        self.assertFalse(factor.requires_fit)
        self.assertEqual(factor.full_name, 'factor@unit_factor')
        self.assertEqual(ModelPath('nn@gru@unit').requires_fit, True)
        with self.assertRaisesRegex(ValueError, 'schedule'):
            ModelConfig(module='linear', override={'model.name': 'linear_unit',
                        'input.type': 'final', 'input.final.types': ['pred@a']}, short_test=False)

    def test_normal_equal_defaults_and_duplicate_schedule(self):
        self.definition['input'].pop('final.weights')
        self.definition['input'].pop('final.precursor')
        self.save_schedule()
        cfg = self.config()
        self.assertEqual(cfg.input_final_precursor, 'normal')
        self.assertEqual(cfg.input_final_weights, 'equal')
        duplicate_root = self.root / 'current'
        duplicate_root.mkdir()
        (duplicate_root / self.schedule.name).write_text(self.schedule.read_text())
        with patch.object(PATH, 'sched', duplicate_root), self.assertRaisesRegex(ValueError, 'Duplicate schedule'):
            ScheduleConfig.find_path(None, 'linear_unit')

    def test_predictor_has_no_checkpoint_and_preserves_single_stock_axis(self):
        model = get_predictor_module(self.config()).load_model()
        self.assertEqual(model.forward([torch.tensor([[[2.]]]), torch.tensor([[[4.]]])]).tolist(), [[6.]])
        self.assertEqual(model.forward(torch.tensor([[[2.]]])).shape, (1, 1))
        with self.assertRaises(RuntimeError):
            model.fit()

    def test_schedule_required_collision_and_explicit_resume(self):
        self.assertEqual(ScheduleConfig.find_path(None, 'linear_unit'), self.schedule)
        with self.assertRaises(FileNotFoundError):
            self.config(resume=1)
        cfg = self.config()
        marker = cfg.base_path.snapshot('pred_recorder', 'keep.txt')
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('keep')
        self.assertTrue(self.config(resume=1).is_resuming)
        self.assertTrue(marker.exists())
        self.definition['input']['final.weights'] = [1, -1]
        self.save_schedule()
        with self.assertRaisesRegex(ValueError, 'resume=0'):
            self.config(resume=1)
        self.assertTrue(marker.exists())
        self.config(resume=0)
        self.assertFalse(marker.exists())
        self.schedule.unlink()
        with self.assertRaises(FileNotFoundError):
            ModelConfig(base_path=cfg.base_path, short_test=False, resume=1)

    def test_trainer_pipeline_records_predictions_and_resumes_without_archives(self):
        from src.data.util import DataBlock
        from src.res.model.util.data.module_data import ModuleData
        from src.res.model.model_module.application.trainer import ModelTrainer
        from src.proj import Save, Load
        dates = np.array([20240102, 20240103, 20240104, 20240105])
        secids = np.arange(100, 164)
        values = torch.arange(64, dtype=torch.float32)[:, None, None, None].expand(64, 4, 1, 2).clone()
        labels = DataBlock(values, secids, dates, ['std_lag1_10', 'rtn_lag1_10'])
        def load_blocks(module):
            module.blocks['y'] = labels
        def frame(source, requested, **kwargs):
            self.assertEqual(trainer.status.stage, 'data')
            multiplier = 1 if source.reference == 'a' else 2
            return pd.DataFrame([(secid, date, float(secid * multiplier))
                                 for date in requested for secid in secids],
                                columns=['secid', 'date', 'value'])
        self.definition['model'].update(beg_date=20240102, end_date=20240104)
        self.definition['train'] = {'callbacks': ['BasicTestResult']}
        self.definition['env'] = {'mem_storage': True}
        self.save_schedule()
        with patch.object(ModuleData, 'target_start_end', return_value=(20240102, 20240105)), \
             patch.object(ModuleData, '_load_blocks', load_blocks), \
             patch.object(FinalSource, 'frame', autospec=True, side_effect=frame) as read, \
             patch.object(BaseTrainer, 'main_process', BaseTrainer.main_process.__wrapped__), \
             patch.object(BaseTrainer, 'stage_fit', side_effect=AssertionError('No fit allowed')):
            trainer = ModelTrainer.initialize(schedule_name='linear_unit', short_test=False, vb_level='never')
            trainer.go()
            self.assertEqual(read.call_count, 2)
            Save.async_wait_all()
            paths = list(trainer.record.folder_preds.glob('*.feather'))
            self.assertTrue(paths)
            first = Load.df(paths, key_column=None)
            self.assertEqual(set(first.date), {20240102, 20240103, 20240104})
            np.testing.assert_allclose(first.pred, first.secid * 3)
            self.assertTrue(trainer.base_path.rslt('basic_test.xlsx').is_file())
            self.assertFalse(any(trainer.base_path.archive().rglob('*.pt')))
            self.definition['model']['end_date'] = 20240105
            self.save_schedule()
            read.reset_mock()
            trainer = ModelTrainer.initialize(schedule_name='linear_unit', short_test=False, vb_level='never')
            trainer.go(resume=1)
            Save.async_wait_all()
            second = Load.df(list(trainer.record.folder_preds.glob('*.feather')), key_column=None)
            self.assertEqual(set(second.date), set(dates))
            self.assertEqual(len(second), len(dates) * len(secids))
            np.testing.assert_allclose(second.pred, second.secid * 3)
            self.assertTrue(trainer.config.is_resuming)
            self.assertEqual(read.call_count, 2)
            for call in read.call_args_list:
                self.assertEqual(list(call.args[1]), [20240105])

            # No new period is a successful no-op, including unchanged records.
            read.reset_mock()
            trainer = ModelTrainer.initialize(schedule_name='linear_unit', short_test=False, vb_level='never')
            trainer.go(resume=1)
            Save.async_wait_all()
            unchanged = Load.df(list(trainer.record.folder_preds.glob('*.feather')), key_column=None)
            self.assertEqual(len(unchanged), len(second))
            read.assert_not_called()

            # A hole before the last saved date is the only date read on resume.
            for path in trainer.record.folder_preds.glob('*.feather'):
                frame_on_disk = Load.df(path).query('date != 20240103')
                Save.df(frame_on_disk, path, overwrite=True, vb_level='never')
            read.reset_mock()
            trainer = ModelTrainer.initialize(schedule_name='linear_unit', short_test=False, vb_level='never')
            trainer.go(resume=1)
            Save.async_wait_all()
            self.assertEqual(read.call_count, 2)
            for call in read.call_args_list:
                self.assertEqual(list(call.args[1]), [20240103])
            repaired = Load.df(list(trainer.record.folder_preds.glob('*.feather')), key_column=None)
            self.assertEqual(len(repaired), len(second))
            np.testing.assert_allclose(repaired.pred, repaired.secid * 3)

            # Entirely absent inputs fail before summary/success callbacks.
            from contextlib import nullcontext
            from src.res.model.util import training_history as history
            read.side_effect = lambda *args, **kwargs: pd.DataFrame(columns=['secid', 'date', 'value'])
            trainer = ModelTrainer.initialize(schedule_name='linear_unit', short_test=False, vb_level='never')
            with patch.object(FinalSource, 'storage_paths', return_value=[self.root]), \
                 patch.object(PATH, 'lc_machine', self.root), \
                 patch.object(history, 'git', return_value='test-git'), \
                 patch('src.proj.util.script.code_update_lock.training_code_guard', return_value=nullcontext()), \
                 patch.object(BaseTrainer, 'main_process', history.record_training(BaseTrainer.main_process)), \
                 history.collect_training_runs() as runs, \
                 patch.object(BaseTrainer, 'stage_summary') as summary, \
                 self.assertRaisesRegex(ValueError, 'no finite values'):
                trainer.go(resume=0)
            summary.assert_not_called()
            self.assertEqual(runs[0]['status'], 'failed')

    def test_empty_month_is_skipped_before_nonempty_month(self):
        from src.data.util import DataBlock
        from src.res.model.util.data.module_data import ModuleData
        from src.res.model.model_module.application.trainer import ModelTrainer
        from src.proj import Save, Load
        dates = np.array([20240130, 20240131, 20240201, 20240202])
        secids = np.arange(100, 164)
        values = torch.arange(64, dtype=torch.float32)[:, None, None, None].expand(64, 4, 1, 2).clone()
        labels = DataBlock(values, secids, dates, ['std_lag1_10', 'rtn_lag1_10'])
        def load_blocks(module):
            module.blocks['y'] = labels
        def frame(source, requested, **kwargs):
            self.assertEqual(trainer.status.stage, 'data')
            return pd.DataFrame([(secid, date, float(secid)) for date in requested if date >= 20240201
                                 for secid in secids], columns=['secid', 'date', 'value'])
        self.definition['model'].update(beg_date=20240130, end_date=20240202)
        self.definition['train'] = {'callbacks': ['BasicTestResult']}
        self.definition['env'] = {'mem_storage': True}
        self.save_schedule()
        with patch.object(ModuleData, 'target_start_end', return_value=(20240130, 20240202)), \
             patch.object(ModuleData, '_load_blocks', load_blocks), \
             patch.object(FinalSource, 'frame', autospec=True, side_effect=frame) as read, \
             patch.object(BaseTrainer, 'main_process', BaseTrainer.main_process.__wrapped__):
            trainer = ModelTrainer.initialize(schedule_name='linear_unit', short_test=False, vb_level='never')
            trainer.go()
            Save.async_wait_all()
            result = Load.df(list(trainer.record.folder_preds.glob('*.feather')), key_column=None)
            self.assertEqual(set(result.date), {20240201, 20240202})
            self.assertEqual(trainer.data._final_loader.coverage_by_date[20240130], 0)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(trainer.data.test_full_dates.tolist(), [20240201, 20240202])

    def test_partial_month_and_late_component_keep_batch_dates_aligned(self):
        from src.data.util import DataBlock
        from src.res.model.util.data.module_data import ModuleData
        from src.res.model.model_module.application.trainer import ModelTrainer
        from src.proj import Save, Load
        dates = np.array([20240102, 20240103, 20240104, 20240105, 20240108, 20240109])
        secids = np.arange(100, 164)
        values = torch.arange(64, dtype=torch.float32)[:, None, None, None].expand(64, len(dates), 1, 2).clone()
        labels = DataBlock(values, secids, dates, ['std_lag1_10', 'rtn_lag1_10'])
        def load_blocks(module):
            module.blocks['y'] = labels
        def frame(source, requested, **kwargs):
            self.assertEqual(trainer.status.stage, 'data')
            valid = [20240103, 20240105, 20240108] if source.reference == 'a' else [20240108]
            multiplier = 1 if source.reference == 'a' else 3
            return pd.DataFrame([(secid, date, float(secid * multiplier)) for date in requested if date in valid
                                 for secid in secids], columns=['secid', 'date', 'value'])
        self.definition['model'].update(beg_date=20240102, end_date=20240109)
        self.definition['input']['final.weights'] = 'equal'
        self.definition['train'] = {'callbacks': ['BasicTestResult']}
        self.definition['env'] = {'mem_storage': True}
        self.save_schedule()
        with patch.object(ModuleData, 'target_start_end', return_value=(20240102, 20240109)), \
             patch.object(ModuleData, '_load_blocks', load_blocks), \
             patch.object(FinalSource, 'frame', autospec=True, side_effect=frame) as read, \
             patch.object(BaseTrainer, 'main_process', BaseTrainer.main_process.__wrapped__):
            trainer = ModelTrainer.initialize(schedule_name='linear_unit', short_test=False, vb_level='never')
            trainer.go()
            Save.async_wait_all()
            result = Load.df(list(trainer.record.folder_preds.glob('*.feather')), key_column=None)
            self.assertEqual(set(result.date), {20240103, 20240105, 20240108})
            np.testing.assert_allclose(result.pred, result.secid * np.where(result.date == 20240108, 2, 1))
            self.assertEqual(trainer.data.test_full_dates.tolist(), [20240103, 20240105, 20240108])
            self.assertEqual(trainer.batch_dates.tolist(), [20240103, 20240105, 20240108])
            self.assertEqual(len(trainer.data.model_date_list), 1)
            self.assertEqual(read.call_count, 2)
            # Known empty dates also count as checked; a no-op resume reads no sources.
            read.reset_mock()
            trainer = ModelTrainer.initialize(schedule_name='linear_unit', short_test=False, vb_level='never')
            trainer.go(resume=1)
            read.assert_not_called()
            Save.async_wait_all()
            self.assertEqual(trainer.data.test_full_dates.tolist(), [20240103, 20240105, 20240108])

    def test_sample_schedules_parse(self):
        real_schedules = Path(__file__).resolve().parents[1] / 'configs/schedule/linear'
        for name, weights, precursor in [('gru_only', 'equal', 'normal'),
                                         ('gru_df_equal', 'equal', 'normal'),
                                         ('gru_df_weighted', [0.7, 0.3], 'normal'),
                                         ('gru_df_raw_sum', [1, 1], 'none')]:
            (PATH.sched_linear / f'{name}.yaml').write_text((real_schedules / f'{name}.yaml').read_text())
            cfg = ModelConfig(schedule_name=name, short_test=False, vb_level='never')
            self.assertEqual(cfg.base_path.full_name, f'linear@{name}')
            self.assertEqual(cfg.input_final_weights, weights)
            self.assertEqual(cfg.input_final_precursor, precursor)
            self.assertEqual(cfg.input_final_types, ['pred@gru_day_V1'] +
                             ([] if name == 'gru_only' else ['factor@df_scores_v0']))

    def test_go_resume_is_forwarded_before_setup(self):
        trainer = object.__new__(BaseTrainer)
        trainer._kwargs = {}
        with patch.object(BaseTrainer, 'main_process', return_value='ran'):
            self.assertEqual(trainer.go(resume=1), 'ran')
        self.assertEqual(trainer._kwargs['resume'], 1)

    def test_preflight_failure_preserves_previous_results_and_zero_weight_skips(self):
        cfg = self.config()
        marker = cfg.base_path.snapshot('pred_recorder', 'keep.txt')
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('previous results')
        self.source_validation.side_effect = FileNotFoundError('missing required source')
        with self.assertRaisesRegex(FileNotFoundError, 'missing required source'):
            self.config()
        self.assertTrue(marker.exists())
        self.source_validation.side_effect = None
        self.source_validation.reset_mock()
        self.definition['input']['final.weights'] = [1, 0]
        self.save_schedule()
        self.config()
        self.assertEqual(self.source_validation.call_count, 1)


if __name__ == '__main__':
    unittest.main()
