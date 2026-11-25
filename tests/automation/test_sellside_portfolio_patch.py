"""Verify portfolio replay scope, continuity and recoverability without production writes."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.automation.version_patch import load_patch_support


patch_module = load_patch_support('0.7.11', 'sellside_portfolios')


class SellsidePortfolioPatchTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.ctx = SimpleNamespace(root=self.root, backup_dir=self.root / 'backup')
        self.ctx.backup_dir.mkdir()
        self.enterContext(patch.object(patch_module.PATH, 'production', self.root))
        self.enterContext(patch.object(patch_module.PATH, 'rslt_trade', self.root / 'results/trade'))
        self.enterContext(patch.object(patch_module.PATH, 'trade_port', self.root / 'data/Export/trading_portfolio'))
        self.enterContext(patch.object(patch_module.MACHINE, 'platform_server', True))
        self.enterContext(patch.object(patch_module.CALENDAR, 'updated', return_value=20260108))
        self.enterContext(patch.object(patch_module, '_check_vendor_history'))
        self.publish = self.enterContext(patch.object(patch_module, '_publish'))
        self.enterContext(patch.object(patch_module.TrackingPort, 'candidate_ports', {
            'use_daily': {'alpha': ['factor@df_scores_v3', 'gru_day_V1']},
            'gru_daily': {'alpha': 'gru_day_V1'},
        }))
        self.enterContext(patch.object(patch_module.BacktestPort, 'candidate_ports', {
            'use_daily.backtest': {'alpha': 'blend', 'components': ['factor@df_scores_v0', 'gru_day_V1']},
        }))

    @staticmethod
    def frame(secid):
        return pd.DataFrame({'secid': [secid], 'weight': [1.], 'value': [100.]})

    def holding(self, port, date, secid):
        path = port.export_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame(secid).to_feather(path)
        return path

    def test_replays_existing_outputs_with_prior_holdings_and_preserves_other_ports(self):
        tracking = patch_module.TrackingPort.load('use_daily')
        unrelated = patch_module.TrackingPort.load('gru_daily')
        backtest = patch_module.BacktestPort.load('use_daily.backtest')
        prefix = self.holding(tracking, 20251231, 1)
        backtest_prefix = self.holding(backtest, 20251231, 8)
        old_paths = [self.holding(tracking, date, 99) for date in (20260105, 20260107)]
        old_paths.append(self.holding(backtest, 20260107, 99))
        unrelated_path = self.holding(unrelated, 20260107, 20)
        original = {path: path.read_bytes() for path in (prefix, backtest_prefix, unrelated_path, *old_paths)}

        csv = patch_module.PATH.rslt_trade / 'trading_ports/trading_ports.20260107.csv'
        csv.parent.mkdir(parents=True)
        pd.DataFrame({'name': ['use_daily', 'gru_daily'], 'date': [20260107, 20260107],
                      'secid': [99, 20], 'weight': [1., 1.], 'value': [100., 100.]}).to_csv(csv)

        predecessors = []

        def track(port, date, *, last_port, export):
            predecessor = int(last_port.to_dataframe().secid.iloc[0])
            predecessors.append(predecessor)
            frame = self.frame(predecessor + 1).assign(name=port.name, date=date)
            port.save_port(frame, date)
            return frame

        def simulate(port, end):
            self.assertEqual(port.end_date(), 20260106)
            seed = port.get_last_port(end).to_dataframe().secid.iloc[0]
            port.save_port(self.frame(seed + 1), end)

        def analyze(port, **kwargs):
            self.assertFalse(port.result_path_account.exists())
            port.result_dir.mkdir(parents=True, exist_ok=True)
            port.result_path_account.write_bytes(b'rebuilt account')

        with patch.object(patch_module.StagedTrackingPort, 'build_portfolio', track), \
                patch.object(patch_module.StagedBacktestPort, 'build_backward', simulate), \
                patch.object(patch_module.StagedTrackingPort, 'analyze', analyze), \
                patch.object(patch_module.StagedBacktestPort, 'analyze', analyze), \
                patch('src.res.factor.util.classes.alpha_model.AlphaComposite.get_alpha_models'):
            patch_module.apply(self.ctx)
            patch_module.apply(self.ctx)

        self.assertEqual(predecessors, [1, 2])
        self.assertEqual([int(pd.read_feather(path).secid.iloc[0]) for path in old_paths], [2, 3, 9])
        for path in (prefix, backtest_prefix, unrelated_path):
            self.assertEqual(path.read_bytes(), original[path])
        for path in old_paths:
            self.assertEqual((self.ctx.backup_dir / 'original' / path.relative_to(self.root)).read_bytes(),
                             original[path])
        self.assertEqual(pd.read_csv(csv, index_col=0).set_index('name').secid.to_dict(),
                         {'use_daily': 3, 'gru_daily': 20})
        self.assertTrue(patch_module.verify(self.ctx))
        self.publish.assert_called_once()
        old_paths[0].write_bytes(b'corrupt')
        self.assertFalse(patch_module.verify(self.ctx))

    def test_missing_rebalanced_date_preserves_originals(self):
        port = patch_module.TrackingPort.load('use_daily')
        path = self.holding(port, 20260107, 99)
        original = path.read_bytes()
        with patch.object(patch_module.StagedTrackingPort, 'build_portfolio', return_value=pd.DataFrame()), \
                patch('src.res.factor.util.classes.alpha_model.AlphaComposite.get_alpha_models'):
            with self.assertRaisesRegex(ValueError, 'produced no holdings'):
                patch_module.apply(self.ctx)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(patch_module.verify(self.ctx))
        self.publish.assert_not_called()

    def test_receiver_leaves_locally_synced_data_in_place(self):
        with patch.object(patch_module.MACHINE, 'platform_server', False):
            patch_module.apply(self.ctx)
            self.assertTrue(patch_module.verify(self.ctx))
        self.assertEqual(json.loads((self.ctx.backup_dir / patch_module.JOURNAL).read_text())['status'], 'receiver')
        self.publish.assert_not_called()

    def test_interrupted_install_keeps_all_originals_for_retry(self):
        port = patch_module.TrackingPort.load('use_daily')
        outputs = {}
        for date in (20260105, 20260107):
            target = self.holding(port, date, 99)
            staged = self.root / f'new-{date}.feather'
            self.frame(7).to_feather(staged)
            outputs[target] = staged
        last_target = next(reversed(outputs))
        copy = patch_module._atomic_copy

        def interrupt(source, target, **kwargs):
            if target == last_target:
                raise OSError('interrupted install')
            copy(source, target, **kwargs)

        with patch.object(patch_module, '_atomic_copy', side_effect=interrupt):
            with self.assertRaisesRegex(OSError, 'interrupted install'):
                patch_module._install(outputs, self.ctx, self.root)
        self.assertFalse(patch_module.verify(self.ctx))
        patch_module._install(outputs, self.ctx, self.root)
        for target in outputs:
            backup = self.ctx.backup_dir / 'original' / target.relative_to(self.root)
            self.assertEqual(pd.read_feather(backup).secid.tolist(), [99])
            self.assertEqual(pd.read_feather(target).secid.tolist(), [7])
        self.assertTrue(patch_module.verify(self.ctx))
