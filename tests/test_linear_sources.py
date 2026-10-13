"""Shared arithmetic and cross-machine/whole-period final-source failures."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from src.proj import PATH
from src.func.linear import weighted_components
from src.res.factor.util.classes.alpha_model import Amodel
from src.res.model.util.final_source import FinalSource, FinalInputLoader, prepare_final_values


class SharedLinearTest(unittest.TestCase):
    def test_daily_empty_components_equal_and_literal_weights(self):
        raw = torch.tensor([[[float('nan'), float('nan')], [2., float('nan')], [2., 6.]],
                            [[float('nan'), float('nan')], [4., float('nan')], [4., 8.]]])
        equal = prepare_final_values(raw, 'none', 'equal').sum(-1)
        self.assertTrue(equal[:, 0].isnan().all())
        torch.testing.assert_close(equal[:, 1:], torch.tensor([[2., 4.], [4., 6.]]))
        literal = prepare_final_values(raw, 'none', [.7, .3]).sum(-1)
        torch.testing.assert_close(literal[:, 1:], torch.tensor([[1.4, 3.2], [2.8, 5.2]]))
        normal = prepare_final_values(raw, 'normal', 'equal').sum(-1)
        torch.testing.assert_close(normal[:, 1:], torch.tensor([[-1., -1.], [1., 1.]]))
        percentile = prepare_final_values(raw, 'percentile', 'equal').sum(-1)
        torch.testing.assert_close(percentile[:, 1:], torch.tensor([[.5, .5], [1., 1.]]))

    def test_composite_union_weights_normalization_and_single_component(self):
        a = Amodel(20240102, np.array([2., 6.]), np.array([1, 2]), 'a')
        b = Amodel(20240102, np.array([4., 8.]), np.array([2, 3]), 'b')
        raw = Amodel.combine_linear([a, b], [2, 1], normalize=False)
        np.testing.assert_array_equal(raw.secid, [1, 2, 3])
        np.testing.assert_allclose(raw.alpha, [4/3, 16/3, 8/3])
        normalized = Amodel.combine_linear([a, b], [2, 1])
        np.testing.assert_allclose(normalized.alpha, (raw.alpha - raw.alpha.mean()) / raw.alpha.std())
        self.assertIs(Amodel.combine_linear([a], [7], normalize=False), a)
        np.testing.assert_allclose(Amodel.combine_linear([a], [7]).alpha, [-1, 1])
        np.testing.assert_allclose(Amodel.combine_linear([a, b], [2, -1], normalize=False).alpha, [4, 8, -8])

    def test_final_and_composite_agree_on_aligned_values(self):
        values = np.array([[2., 4.], [6., 8.], [9., 3.]])
        alphas = [Amodel(20240102, values[:, i], np.arange(3)) for i in range(2)]
        for weights in ('equal', [0.7, 0.3]):
            final = prepare_final_values(torch.tensor(values[:, None, :]), 'none', weights).sum(-1)
            composite = Amodel.combine_linear(alphas, None if weights == 'equal' else weights, normalize=False)
            np.testing.assert_allclose(final[:, 0], composite.alpha)
        np.testing.assert_allclose(weighted_components(values, [1, -1]).sum(-1), [-2, -2, 6])
        with self.assertRaisesRegex(ValueError, 'nonzero sum'):
            weighted_components(values, [1, -1], normalize_weights=True)


class FinalSourceAvailabilityTest(unittest.TestCase):
    def test_missing_pred_and_model_paths_have_source_context(self):
        with tempfile.TemporaryDirectory() as root, \
             patch.object(PATH, 'pred', Path(root)), patch.object(PATH, 'model_nn', Path(root)):
            with self.assertRaisesRegex(FileNotFoundError, 'pred@missing.*20240102'):
                FinalSource.parse('pred@missing').frame([20240102])
            with self.assertRaisesRegex(FileNotFoundError, 'model_pred@gru@missing.*20240102'):
                FinalSource.parse('model_pred@gru@missing').validate('20240102..20240103')
        with self.assertRaisesRegex(ValueError, 'Unknown factor.*factor@unknown_linear_probe'):
            FinalSource.parse('factor@unknown_linear_probe').validate()

    def test_affiliate_alternative_storage_without_generic_factor_directory(self):
        from src.res.factor.calculator import AffiliateFactorCalculator
        class ExampleFactor(AffiliateFactorCalculator):
            load_db_src = 'sellside'
            load_db_key = 'primary'
            alternative_dbs = ({'src': 'sellside', 'key': 'backup', 'col': 'score'},)
        source = FinalSource.parse('factor@availability_probe')
        with tempfile.TemporaryDirectory() as root, \
             patch.object(PATH, 'db', Path(root)), \
             patch.object(PATH, 'factor', Path(root) / 'absent_generic_factor'), \
             patch.object(FinalSource, 'factor_calculator', return_value=ExampleFactor):
            with self.assertRaisesRegex(FileNotFoundError, 'primary.*backup'):
                source.validate()
            (Path(root) / 'DB_sellside' / 'backup').mkdir(parents=True)
            source.validate()
            frame = pd.DataFrame({'secid': [1], 'date': [20240102], 'score': [3.]})
            with patch.object(ExampleFactor, 'Loads', return_value=frame) as load:
                self.assertEqual(source.frame([20240102]).value.tolist(), [3.])
                self.assertEqual(load.call_args.kwargs, {'normalize': False, 'closest': False})

    def loader(self):
        return FinalInputLoader(SimpleNamespace(input_final_types=['pred@a', 'pred@b'],
            input_final_weights='equal', input_final_precursor='none', input_final_refresh=False))

    def test_empty_month_then_valid_month_and_skipped_summary(self):
        loader = self.loader()
        empty = pd.DataFrame(columns=['secid', 'date', 'value'])
        full = pd.DataFrame({'secid': [1], 'date': [20240201], 'value': [2.]})
        with patch.object(FinalSource, 'frame', return_value=empty):
            loader.blocks([20240102], [1], [20240102, 20240201])
        with patch.object(FinalSource, 'frame', return_value=full):
            loader.blocks([20240201], [1], [20240102, 20240201])
        summary = loader.validate_coverage([20240102, 20240201])
        self.assertIn('1/2 usable dates', summary)
        self.assertIn('skipped dates: [20240102]', summary)
        self.assertIn('no new dates', loader.validate_coverage([]))

    def test_entirely_empty_component_is_ignored_when_another_has_data(self):
        loader = self.loader()
        def frame(source, dates, **kwargs):
            rows = [(1, 20240102, 4.)] if source.reference == 'a' else []
            return pd.DataFrame(rows, columns=['secid', 'date', 'value'])
        with patch.object(FinalSource, 'frame', autospec=True, side_effect=frame):
            blocks = loader.blocks([20240102], [1], [20240102])
        self.assertEqual(blocks['pred@a'].values.item(), 4.)
        self.assertEqual(blocks['pred@b'].values.item(), 0.)
        self.assertIn("ignored empty components: ['pred@b']", loader.validate_coverage([20240102]))

    def test_empty_nan_disjoint_and_future_only_fail_at_period_end(self):
        for kind in ('empty', 'nan', 'disjoint', 'future'):
            with self.subTest(kind=kind), patch.object(FinalSource, 'storage_paths', return_value=[Path('/source')]):
                loader = self.loader()
                if kind == 'future':
                    from src.res.model.model_module.application.predictor import ArchivedPredictorModel
                    loader.config.input_final_types = ['model_pred@gru@future']
                    loader = FinalInputLoader(loader.config)
                    source = Mock(model_dates=np.array([20250101]))
                    with patch.object(ArchivedPredictorModel, 'from_model_reference', return_value=source):
                        loader.blocks([20240102], [1, 2], [20240102])
                    source.pred_values.assert_not_called()
                else:
                    def frame(source, dates, **kwargs):
                        rows = [] if kind == 'empty' else [(1 if source.reference == 'a' or kind != 'disjoint' else 2,
                                                           20240102, np.nan if kind == 'nan' else 1.)]
                        return pd.DataFrame(rows, columns=['secid', 'date', 'value'])
                    with patch.object(FinalSource, 'frame', autospec=True, side_effect=frame):
                        loader.blocks([20240102], [1, 2], [20240102])
                with self.assertRaisesRegex(ValueError, 'no common finite stocks' if kind == 'disjoint' else 'no finite values'):
                    loader.validate_coverage([20240102])


if __name__ == '__main__':
    unittest.main()
