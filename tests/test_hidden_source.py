"""Hidden input selection is exact in interactive and headless training alike."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.proj import PATH
from src.res.model.model_module.application.predictor import ArchivedPredictorModel
from src.res.model.util.config import ModelConfig
from src.res.model.util.core import ModelPath


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
