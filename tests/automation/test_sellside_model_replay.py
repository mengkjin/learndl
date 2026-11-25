import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.automation.version_patch import load_patch_support

repair = load_patch_support('0.7.14', 'sellside_tests')


class SellsideModelReplayTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.ctx = SimpleNamespace(root=self.root, backup_dir=self.root / 'backup')
        self.ctx.backup_dir.mkdir()
        for attr, value in [('production', self.root), ('model', self.root/'models'),
                            ('model_factor', self.root/'models/factor'), ('model_linear', self.root/'models/linear')]:
            self.enterContext(patch.object(repair.PATH, attr, value))
        self.enterContext(patch.object(repair.MACHINE, 'platform_server', True))
        self.enterContext(patch.object(repair.CALENDAR, 'updated', return_value=20261125))
        self.enterContext(patch.object(repair, 'load_patch_support'))
        self.base = repair.PATH.model_factor/'df_scores_v0'
        (self.base/'configs').mkdir(parents=True)
        (self.base/'configs/model.yaml').write_text('model.end_date: 99991231')
        self.outputs(self.base, [20251231, 20260105], b'original')

    def outputs(self, base, dates, content):
        for rel in ['snapshot/pred_recorder/avg_preds', 'snapshot/basic_test', 'results']:
            (base / rel).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'date': dates}).to_feather(base/'snapshot/pred_recorder/avg_preds/record.feather')
        (base/'snapshot/basic_test/test_summary.feather').write_bytes(content)
        (base/'results/detailed_alpha_data.xlsx').write_bytes(content)

    def test_replaces_only_after_full_history_and_handles_interrupted_install(self):
        (self.base/'snapshot/detailed_alpha/screen_fmp_test').mkdir(parents=True)
        def run(*, base_path, resume, override, **kwargs):
            self.assertIn('screen', override['callbacks.DetailedAlphaAnalysis']['tasks'])
            self.assertEqual(resume, 0)
            self.assertFalse((base_path/'snapshot').exists())
            self.outputs(base_path, [20251231, 20260105, 20261125], b'new')

        with patch.object(repair, 'run_staged_test', side_effect=run) as go:
            original = Path.rename
            def crash(path, target):
                if path.name == 'results' and path.parent.name == 'new':
                    raise RuntimeError('interrupted')
                return original(path, target)
            with patch.object(Path, 'rename', crash), self.assertRaisesRegex(RuntimeError, 'interrupted'):
                repair.apply(self.ctx)
            repair.apply(self.ctx)
            go.assert_called_once()
        self.assertTrue(repair.verify(self.ctx))
        self.assertEqual(next(repair.PATH.model.rglob('original/results/detailed_alpha_data.xlsx')).read_bytes(), b'original')
        self.assertEqual((self.base/'results/detailed_alpha_data.xlsx').read_bytes(), b'new')

    def test_legacy_holdings_and_account_require_both_current_files(self):
        relative = Path('snapshot/detailed_alpha/t50_fmp_test')
        files = ['portfolio/top.best.univ.top_50.lag0.feather', 'account/top.best.univ.top_50.lag0.tar']
        for name in files:
            path = self.base / relative / name
            path.parent.mkdir(parents=True)
            path.write_bytes(b'old')
        def run(*, base_path, **kwargs):
            self.outputs(base_path, [20251231, 20260105], b'new')
            for name in rebuilt:
                path = base_path / relative / repair.FMP_RESUME_VERSION / name
                path.parent.mkdir(parents=True)
                path.write_bytes(b'rebuilt')
        with patch.object(repair, 'run_staged_test', side_effect=run):
            rebuilt = files[:1]
            with self.assertRaisesRegex(ValueError, 'missing rebuilt analysis.*account'):
                repair.apply(self.ctx)
            rebuilt = files
            repair.apply(self.ctx)
        self.assertTrue(repair.verify(self.ctx))
        self.assertFalse((self.base / relative / files[0]).exists())

    def test_replay_accepts_group_files_written_by_factor_calculators(self):
        from src.res.factor.analytic.factor_perf import Group_Percentile, Group_Return, IC_Monotony, FactorPerfTest
        from src.res.factor.util import Benchmark, StockFactor

        groups = Path('snapshot/detailed_alpha/factor_perf_test/factor_stats/group_perf')
        (self.base / groups).mkdir(parents=True)
        dates = [20251231, 20260105]
        frame = pd.DataFrame([(date, secid, float(secid)) for date in dates for secid in range(100)],
                             columns=['date', 'secid', 'best'])
        factor = StockFactor(frame, factor_names=['best'])
        with patch.object(Benchmark, 'factor_mask', side_effect=lambda data: data), \
                patch.object(StockFactor, 'frame_with_cols', return_value=factor.frame().assign(ret=0.01)):
            for task in (IC_Monotony(), Group_Return(), Group_Percentile()):
                task.calc(factor)

        def run(*, base_path, **kwargs):
            self.outputs(base_path, dates, b'new')
            factor.cache_factor_stats.save(base_path / groups.parent)
            if omit_rank:
                next((base_path / groups).glob('*rank=True.feather')).unlink()

        with patch.object(repair, 'run_staged_test', side_effect=run):
            omit_rank = True
            with self.assertRaisesRegex(ValueError, 'missing rebuilt group statistics'):
                repair.apply(self.ctx)
            omit_rank = False
            repair.apply(self.ctx)
        required = FactorPerfTest.required_group_stats()
        self.assertEqual({p['bm'] for p in required}, {'None', 'csi300', 'csi500', 'csi1000'})
        self.assertTrue(repair.verify(self.ctx))

    def test_incomplete_replay_preserves_original_output_and_no_receipt(self):
        def run(*, base_path, **kwargs):
            self.outputs(base_path, [20260105], b'incomplete')
        with patch.object(repair, 'run_staged_test', side_effect=run), self.assertRaisesRegex(ValueError, 'incomplete'):
            repair.apply(self.ctx)
        self.assertEqual((self.base/'results/detailed_alpha_data.xlsx').read_bytes(), b'original')
        self.assertFalse(repair.verify(self.ctx))
