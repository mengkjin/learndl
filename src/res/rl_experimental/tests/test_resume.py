import importlib
import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.res.rl_experimental.bundle import _sha256
from src.res.rl_experimental.progress import quiet_output
from src.res.rl_experimental.recover_results import audit_suite
from src.res.rl_experimental.resume import git_sources, historical_package, inspect_resume, resume_suite
from src.res.rl_experimental.tests.test_walk_forward import real_like_panel


class ResumeTest(unittest.TestCase):
    def test_cross_revision_resume_only_missing_task_and_replay(self):
        self._run_case()

    def test_actual_20261026_legacy_engine(self):
        revision = '858dca6bd47070dcc0960dc8689542e987c2d320'
        try:
            sources = git_sources(Path(__file__).parents[4], revision)
        except subprocess.CalledProcessError:
            self.skipTest('Historical regression source not present in this shallow clone')
        self._run_case(sources)

    def _run_case(self, sources=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root/'src/res/rl_experimental'
            source.mkdir(parents=True)
            if sources is None:
                sources = {path.name: path.read_bytes() for path in Path(__file__).parents[1].glob('*.py')}
            for name, data in sources.items():
                (source/name).write_bytes(data)

            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True,
                                      capture_output=True, text=True).stdout.strip()

            git('init', '-q')
            git('config', 'user.email', 'test@example.invalid')
            git('config', 'user.name', 'Test')
            git('add', 'src')
            git('commit', '-qm', 'original research code')
            revision = git('rev-parse', 'HEAD')
            original_sources = git_sources(root, revision)
            self.assertIn('walk_forward.py', original_sources)
            panel = real_like_panel()
            panel.save_npz(root/'panel.npz')
            run = root/'results/rl_experimental/run'
            config = dict(panel=str(root/'panel.npz'), output=str(run),
                screen=dict(start=20170101, valid_start=20210101, end=20230101),
                first_quarter='2023Q1', last_quarter='2023Q2', seeds=[7], penalties=[.001], bonuses=[0, .5],
                costs_bp=[3, 5], screen_steps=16, quarter_steps=16,
                training=dict(device='cpu', n_envs=1, rollout_steps=8, batch_size=8, update_epochs=1, hidden_dim=8))
            config_path = root/'initial.json'
            config_path.write_text(json.dumps(config))
            with quiet_output(root/'test.log'), historical_package(source) as engine:
                experiment = importlib.import_module('src.res.rl_experimental.experiment')
                original_info = experiment._system_info()
                with patch.object(experiment, '_system_info', return_value=original_info | {'git_revision': revision}):
                    engine.run_walk_forward(config_path)
            self.assertTrue(audit_suite(run)['complete'])
            # New logging/source code must not force completed models to retrain.
            (source/'progress.py').write_text('# changed logging\n'+(source/'progress.py').read_text())
            git('add', 'src')
            git('commit', '-qm', 'logging only')
            kept = run/'quarters/2023Q1/original/seed-7/best_model.zip'
            stamp, digest = kept.stat().st_mtime_ns, _sha256(kept)
            missing = run/'quarters/2023Q2/retention/seed-7'
            failed = json.loads((missing/'task_status.json').read_text())
            shutil.rmtree(missing)
            missing.mkdir()
            (missing/'task_status.json').write_text(json.dumps(failed | {'status': 'failed'}))
            (run/'status.json').write_text(json.dumps({'status': 'failed', 'stage': 'all'}))
            plan = inspect_resume(run, root)
            self.assertEqual(len(plan['completed']), 7)
            self.assertEqual(plan['pending'], ['quarters/2023Q2/retention/seed-7'])
            self.assertEqual(plan['source_commit'], revision)
            self.assertEqual(plan['stage'], 'quarters')
            saved_config = (run/'config.json').read_bytes()
            changed = json.loads(saved_config)
            changed['quarter_steps'] = 17
            (run/'config.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                inspect_resume(run, root)
            (run/'config.json').write_bytes(saved_config)
            kept_bytes = kept.read_bytes()
            kept.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                inspect_resume(run, root)
            kept.write_bytes(kept_bytes)
            stamp = kept.stat().st_mtime_ns
            # A changed snapshot must fail before any missing task starts.
            changed_features = panel.features.copy()
            changed_features[0, 0, 0] += 100
            replace(panel, features=changed_features).save_npz(root/'panel.npz')
            with self.assertRaisesRegex(ValueError, 'suite config/data/code changed'):
                resume_suite(run, root, run/'recovery/changed-panel')
            self.assertFalse((missing/'metrics.json').exists())
            panel.save_npz(root/'panel.npz')
            resumed = resume_suite(run, root, run/'recovery/attempt')
            self.assertEqual(len(resumed['pending']), 1)
            self.assertEqual(kept.stat().st_mtime_ns, stamp)
            self.assertEqual(_sha256(kept), digest)
            audit = audit_suite(run)
            self.assertTrue(audit['complete'], audit['errors'])
            self.assertEqual(audit['training_tasks_verified'], 8)
            self.assertEqual(audit['test_paths_verified'], 18)
            # All training exists but evaluation is missing: replay-only recovery.
            (run/'summary/robust_3.csv').unlink()
            self.assertEqual(inspect_resume(run, root)['pending'], [])
            resume_suite(run, root, run/'recovery/replay-only')
            self.assertTrue(audit_suite(run)['complete'])
            self.assertEqual(kept.stat().st_mtime_ns, stamp)


if __name__ == '__main__':
    unittest.main()
