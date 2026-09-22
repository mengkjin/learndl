"""Hidden input selection is exact in interactive and headless training alike."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import torch

from src.proj import PATH
from src.res.model.model_module.application.predictor import ArchivedPredictorModel
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import ModelPath, BatchData, BatchOutput
from src.res.model.util.data.data_module import DataModule


class HiddenSourceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        for key, relative in [('model', 'models'), ('model_nn', 'models/nn'),
                              ('model_boost', 'models/boost'), ('model_st', 'models/st')]:
            folder = self.root / relative
            folder.mkdir(parents=True, exist_ok=True)
            self.enterContext(patch.object(PATH, key, folder))
        self.enterContext(patch.object(ModelPath, 'log_operation'))
        self.prompt = self.enterContext(patch('builtins.input', side_effect=AssertionError('No input allowed')))
        config = ModelConfig(schedule_name='gru_day_new_rtn', stage=1, resume=0, selection=0,
                             short_test=False, vb_level='never').start_model()
        self.base = config.base_path.base
        self.checkpoint = self.base / 'archive/0/20240101/best/state_dict.pt'
        self.checkpoint.parent.mkdir(parents=True)
        torch.save({'weight': torch.ones(1)}, self.checkpoint)

    def load(self, reference='gru@gru_day_new_rtn@1@0@best'):
        return ArchivedPredictorModel.from_model_str(reference)

    def cached_source(self):
        source = self.load()
        source.__dict__['data'] = SimpleNamespace(
            early_test_dates=[], storage=SimpleNamespace(del_group=Mock()))

        def batches(dates, model_date, **kwargs):
            for date in dates:
                inputs = SimpleNamespace(secid=np.array([20, 10]), date=np.array([date, date]), date0=date)
                output = BatchOutput((torch.tensor([[2., 20.], [1., 10.]]),
                                      {'hidden': torch.tensor([[200.], [100.]])}))
                yield BatchData(inputs, output)

        iterator = self.enterContext(patch.object(source, 'iter_batch_data', side_effect=batches))
        return source, iterator

    def test_hidden_predictions_have_separate_caches_and_one_forward_pass(self):
        source, iterator = self.cached_source()
        frame = source.hidden_values([20240102], 20240101, include_pred=True, async_save=False)
        self.assertEqual(iterator.call_count, 1)
        self.assertEqual(iterator.call_args.args, ([20240102], 20240101))
        self.assertEqual(iterator.call_args.kwargs['model_num'], 0)
        self.assertEqual(iterator.call_args.kwargs['submodel'], 'best')
        hidden = pl.read_ipc(source.hidden_values_path(0, 20240101, 'best'))
        pred = pl.read_ipc(source.pred_values_path(0, 20240101, 'best'))
        self.assertEqual(set(hidden.columns), {'secid', 'date', 'hidden.0'})
        self.assertEqual(set(pred.columns), {'secid', 'date', 'pred.0', 'pred.1'})
        self.assertEqual(frame.sort('secid')['pred.0'].to_list(), [1., 2.])
        iterator.reset_mock()
        source.hidden_values([20240102], 20240101, include_pred=True, async_save=False)
        iterator.assert_not_called()

    def test_existing_hidden_is_preserved_when_prediction_cache_is_missing(self):
        source, iterator = self.cached_source()
        source.hidden_values([20240102], 20240101, async_save=False)
        path = source.hidden_values_path(0, 20240101, 'best')
        original = path.read_bytes()
        iterator.reset_mock()
        frame = source.hidden_values([20240102], 20240101, include_pred=True, async_save=False)
        iterator.assert_called_once()
        self.assertEqual(path.read_bytes(), original)
        self.assertIn('pred.1', frame.columns)

    def test_partial_prediction_cache_only_recomputes_missing_dates(self):
        source, iterator = self.cached_source()
        source.hidden_values([20240102, 20240103], 20240101, async_save=False)
        source.pred_values([20240102], 20240101, async_save=False)
        iterator.reset_mock()
        source.hidden_values([20240102, 20240103], 20240101, include_pred=True, async_save=False)
        self.assertEqual(iterator.call_args.args[0], [20240103])
        pred = pl.read_ipc(source.pred_values_path(0, 20240101, 'best'))
        self.assertEqual(pred.height, 4)
        self.assertEqual(pred.select(['secid', 'date']).unique().height, 4)

    def test_hidden_only_does_not_read_or_create_prediction_cache(self):
        source, _ = self.cached_source()
        frame = source.hidden_values([20240102], 20240101, include_pred=False, async_save=False)
        self.assertEqual(set(frame.columns), {'secid', 'date', 'hidden.0'})
        self.assertFalse(source.pred_values_path(0, 20240101, 'best').exists())

    def test_prediction_refresh_preserves_other_dates_and_hidden_file(self):
        source, iterator = self.cached_source()
        source.hidden_values([20240102, 20240103], 20240101, include_pred=True, async_save=False)
        hidden_path = source.hidden_values_path(0, 20240101, 'best')
        original = hidden_path.read_bytes()
        pred_path = source.pred_values_path(0, 20240101, 'best')
        pl.read_ipc(pred_path, memory_map=False).with_columns(pl.lit(-1.).alias('pred.0')).write_ipc(pred_path)
        iterator.reset_mock()
        frame = source.pred_values([20240102], 20240101, load_first=False, async_save=False)
        self.assertEqual(iterator.call_args.args[0], [20240102])
        self.assertEqual(frame.filter(pl.col('date') == 20240102)['pred.0'].to_list(), [1., 2.])
        self.assertEqual(frame.filter(pl.col('date') == 20240103)['pred.0'].to_list(), [-1., -1.])
        self.assertEqual(hidden_path.read_bytes(), original)

    def test_hidden_block_aligns_prediction_features_by_stock_and_date(self):
        source, iterator = self.cached_source()
        source.hidden_values([20240102], 20240101, include_pred=True, async_save=False)
        pred_path = source.pred_values_path(0, 20240101, 'best')
        pl.read_ipc(pred_path, memory_map=False).reverse().write_ipc(pred_path)
        iterator.reset_mock()
        block = source.hidden_block([20240102], 20240101, include_pred=True,
            align_secid=[10, 20], align_date=[20240102], feature_prefix=False, async_save=False)
        iterator.assert_not_called()
        self.assertEqual(set(block.feature), {'hidden.0', 'pred.0', 'pred.1'})
        pred_index = list(block.feature).index('pred.0')
        self.assertEqual(block.values[:, 0, 0, pred_index].tolist(), [1., 2.])

    def test_hidden_prediction_option_defaults_true_and_can_be_disabled(self):
        config = self.load().config
        self.assertTrue(config.input_hidden_include_pred)
        config.model_config.Param['input.special.hidden.include_pred'] = False
        self.assertFalse(config.input_hidden_include_pred)

    def test_empty_request_does_not_infer_or_write(self):
        source, iterator = self.cached_source()
        self.assertTrue(source.hidden_values([], 20240101, include_pred=True).is_empty())
        self.assertTrue(source.pred_values([], 20240101).is_empty())
        iterator.assert_not_called()

    def test_prediction_cache_is_specific_to_checkpoint_date(self):
        source, iterator = self.cached_source()
        source.pred_values([20240102], 20240101, async_save=False)
        iterator.reset_mock()
        source.pred_values([20240102], 20240201, async_save=False)
        self.assertEqual(iterator.call_args.args, ([20240102], 20240201))
        self.assertTrue(source.pred_values_path(0, 20240101, 'best').exists())
        self.assertTrue(source.pred_values_path(0, 20240201, 'best').exists())
        self.assertFalse(source.hidden_values_path(0, 20240101, 'best').exists())

    def test_data_module_appends_scores_only_when_enabled(self):
        source, _ = self.cached_source()
        key = 'gru@gru_day_new_rtn@1@0@best'
        loader = SimpleNamespace(
            input_type='combo', input_keys_hidden=[key],
            y_date=np.array([20240102]), step_idx=np.array([0]), model_date=20240102,
            config=SimpleNamespace(input_hidden_include_pred=False, update_data_param=Mock()),
            datas=SimpleNamespace(secid=np.array([10, 20]), date=np.array([20240102]), x={}))
        original = source.hidden_block
        with patch.object(ArchivedPredictorModel, 'from_model_str', return_value=source), \
             patch.object(source, 'hidden_block', side_effect=lambda *a, **kw: original(*a, **kw, async_save=False)):
            DataModule.setup_loader_inputs_hidden(loader)
            self.assertEqual(len(loader.datas.x[key].feature), 1)
            loader.config.input_hidden_include_pred = True
            DataModule.setup_loader_inputs_hidden(loader)
            self.assertEqual(len(loader.datas.x[key].feature), 3)
            self.assertTrue(any(name.endswith('.pred.1') for name in loader.datas.x[key].feature))
        self.assertEqual(loader.config.update_data_param.call_count, 2)

    def test_multiple_folders_keep_explicit_directory_and_module(self):
        other = self.base.with_name(self.base.name + '@2')
        shutil.copytree(self.base, other)
        # A same-named Boost folder must not affect the explicitly specified GRU.
        shutil.copytree(self.base, PATH.model_boost / 'lgbm@gru_day_new_rtn')
        for ref, expected in [('gru@gru_day_new_rtn@1@0@best', self.base),
                              ('gru@gru_day_new_rtn@2@0@best', other),
                              ('gru@gru_day_new_rtn@0@best', self.base),
                              ('nn@gru@gru_day_new_rtn@2@0@best', other)]:
            with self.subTest(reference=ref):
                model = self.load(ref)
                self.assertEqual(model.path.base, expected)
                self.assertEqual(model.config.base_path.base, expected)
                self.assertEqual(model.config.base_path.model_module, 'gru')
                self.assertEqual(model.path.use_model_nums.tolist(), [0])
        self.prompt.assert_not_called()

    def test_missing_directory_never_falls_back_to_another_index(self):
        self.base.rename(self.base.with_name(self.base.name + '@2'))
        for ref in ('gru@gru_day_new_rtn@1@0@best', 'gru@gru_day_new_rtn@0@best'):
            with self.assertRaisesRegex(FileNotFoundError, 'directory does not exist'):
                self.load(ref)
        self.assertFalse(self.base.exists())
        self.prompt.assert_not_called()

    def test_legacy_number_is_not_a_directory_index(self):
        other = self.base.with_name(self.base.name + '@2')
        shutil.copytree(self.base, other)
        with self.assertRaisesRegex(FileNotFoundError, 'model number 2'):
            self.load('gru@gru_day_new_rtn@2@best')
        self.assertEqual(self.load('gru@gru_day_new_rtn@2@0@best').path.base, other)

    def test_missing_model_number_submodel_or_checkpoint_fails(self):
        with self.assertRaisesRegex(FileNotFoundError, 'model number 1'):
            self.load('gru@gru_day_new_rtn@1@1@best')
        with self.assertRaisesRegex(FileNotFoundError, 'checkpoint does not exist'):
            self.load('gru@gru_day_new_rtn@1@0@swalast')
        self.checkpoint.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'checkpoint does not exist'):
            self.load()
        self.prompt.assert_not_called()

    def test_missing_or_corrupt_saved_config_fails_without_default_fallback(self):
        config = self.base / 'configs/algo.gru.yaml'
        config.write_text('not a mapping')
        with self.assertRaisesRegex(ValueError, 'configuration is corrupt'):
            self.load()
        config.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'configuration is missing'):
            self.load()
        self.prompt.assert_not_called()

    def test_model_dates_belong_to_selected_number(self):
        other = self.base / 'archive/1/20250101/best/state_dict.pt'
        other.parent.mkdir(parents=True)
        torch.save({'weight': torch.ones(1)}, other)
        self.assertEqual(self.load().model_dates.tolist(), [20240101])
        self.assertEqual(self.load('gru@gru_day_new_rtn@1@1@best').model_dates.tolist(), [20250101])

    def test_reference_requires_module_number_and_submodel(self):
        for reference in ('gru_day_new_rtn', 'gru@gru_day_new_rtn', 'gru@gru_day_new_rtn@best',
                          'gru@gru_day_new_rtn@0', 'gru@gru_day_new_rtn@1@all@best',
                          'gru@gru_day_new_rtn@0@0@best', 'gru@gru_day_new_rtn@1@-1@best',
                          'boost@gru@gru_day_new_rtn@1@0@best', 'unknown_module@name@1@0@best'):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.load(reference)
        self.prompt.assert_not_called()


if __name__ == '__main__':
    unittest.main()
