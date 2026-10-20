"""Git-triggered RL attempts, using temporary repositories and no SMTP/GPU."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import portalocker
import yaml

from src.api.task_monitor.scheduling import rl_experiment as rl, idle_worklist as idle, runtime
from src.api.task_monitor.scheduling.recovery import recovery_plan, evidence
from src.proj import PATH
from src.proj.util.script import FitLock, FitLockNN


class RLSchedulingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = runtime.RunStore(self.root / 'state/runs.sqlite')
        self.enterContext(patch.object(rl, 'settings', return_value={'enabled': True, 'interval_seconds': 0}))
        self.enterContext(patch.object(PATH, 'main', self.root))
        self.enterContext(patch.object(idle, 'runtime_dir', return_value=self.root / 'runtime'))
        self.enterContext(patch.object(rl, 'runtime_dir', return_value=self.root / 'runtime'))
        self.enterContext(patch.object(idle, 'idle_reason', return_value=None))
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')
        self.path = self.root / rl.CONFIG_PATH
        self.path.parent.mkdir(parents=True)
        self.doc = dict(enabled=True, mode='single', rerun_mark='one', recipient=None, no_email=False,
                        single=dict(alpha='pred@test', start=20240101, end=20241231,
                                    snapshot_dir='snapshot', output_root=str(self.root / 'results'),
                                    training={'device': 'cpu', 'total_timesteps': 16}),
                        walk_forward=dict(panel='panel.npz', output='results/walk',
                                          screen={}, first_quarter='2023Q1', last_quarter='2023Q2'))
        self.commit()
        self.options = {'enabled': False}

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self):
        self.path.write_text(yaml.safe_dump(self.doc))
        self.git('add', rl.CONFIG_PATH)
        self.git('commit', '-qm', 'config')

    def test_both_formats_and_invalid_values(self):
        self.assertEqual(rl.read_config(self.path)['mode'], 'single')
        self.doc['mode'] = 'walk_forward'
        self.assertEqual(rl.parse_config(yaml.safe_dump(self.doc))['experiment']['kind'], 'walk_forward')
        old = rl.parse_config(json.dumps(self.doc['single']), yaml_format=False)
        self.assertEqual(old['experiment']['alpha'], 'pred@test')
        for update in ({'mode': 'bad'}, {'no_email': 'false'}, {'rerun_mark': []}, {'extra': 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                rl.parse_config(yaml.safe_dump(self.doc | update))
        with self.assertRaises(ValueError):
            rl.parse_config('mode: single\nmode: single')

    def test_git_identity_rerun_revert_and_unrelated_commit(self):
        first = rl.observe(self.store)
        (self.root / 'unrelated').write_text('new')
        self.git('add', 'unrelated')
        self.git('commit', '-qm', 'unrelated')
        self.assertEqual(rl.observe(self.store)['version'], first['version'])
        self.doc['rerun_mark'] = 'two'
        self.commit()
        second = rl.observe(self.store)
        self.assertNotEqual(second['version'], first['version'])
        self.doc['rerun_mark'] = 'one'
        self.commit()
        self.assertNotEqual(rl.observe(self.store)['version'], first['version'])
        self.path.write_text(self.path.read_text() + '# local edit\n')
        self.assertIsNone(rl.observe(self.store))
        self.assertIn('local/uncommitted', rl.status(self.store)['error'])

    def test_failure_and_success_never_retry_after_restart(self):
        for phase in ('error', 'complete', 'killed'):
            request = rl.observe(self.store)
            request.update(id=phase, phase=phase, owner=idle.OWNER)
            self.store.put(request)
            self.assertIsNone(rl.observe(runtime.RunStore(self.store.path)))
            self.assertFalse(recovery_plan(self.store, request, {})['retry'])
            self.doc['rerun_mark'] = phase
            self.commit()
            self.assertIsNotNone(rl.observe(self.store))

    def test_latest_queue_disabled_and_removed(self):
        idle.dispatch(self.options, self.store)
        first = self.store.owned(idle.OWNER)[0]
        self.doc['rerun_mark'] = 'new'
        self.commit()
        idle.dispatch(self.options, self.store)
        rows = self.store.owned(idle.OWNER)
        self.assertEqual(self.store.get(first['id'])['phase'], 'deferred')
        self.assertEqual(sum(r['phase'] == 'queued' for r in rows), 1)
        self.doc['enabled'] = False
        self.commit()
        idle.dispatch(self.options, self.store)
        self.assertFalse(any(r['phase'] == 'queued' for r in self.store.owned(idle.OWNER)))
        self.path.unlink()
        self.assertIsNone(rl.observe(self.store))

    def test_busy_wait_and_same_version_only_one_queue(self):
        with patch.object(idle, 'idle_reason', return_value='GPU busy'):
            idle.dispatch(self.options, self.store)
            self.assertEqual(rl.status(self.store)['wait_reason'], 'GPU busy')
            self.assertEqual(self.store.owned(idle.OWNER), [])
        idle.dispatch(self.options, self.store)
        idle.dispatch(self.options, self.store)
        self.assertEqual(len(self.store.owned(idle.OWNER)), 1)

    def test_observation_interval_does_not_weaken_worker_recheck(self):
        first = rl.observe(self.store)
        self.doc['rerun_mark'] = 'later'
        self.commit()
        with patch.object(rl, 'settings', return_value={'enabled': True, 'interval_seconds': 120}):
            self.assertEqual(rl.observe(self.store, throttle=True)['version'], first['version'])
            self.assertNotEqual(rl.observe(self.store)['version'], first['version'])

    def test_alternating_model_and_rl(self):
        self.store.put(dict(id='last', owner=idle.OWNER, task='model', phase='complete', started_at=1))
        with patch.object(idle, 'worklist_schedules', return_value=({}, {'model': {}}, [])), \
             patch.object(idle, '_candidate', return_value={'task': 'model', 'version': 'm'}):
            idle.dispatch({'enabled': True}, self.store)
            queued = self.store.owned(idle.OWNER)[-1]
            self.assertEqual(queued['task_kind'], rl.TASK)
            queued.update(phase='complete', started_at=2)
            self.store.put(queued)
            self.doc['rerun_mark'] = 'next'
            self.commit()
            idle.dispatch({'enabled': True}, self.store)
            self.assertEqual(self.store.owned(idle.OWNER)[-1]['task'], 'model')

    def test_mail_single_result_suppression_and_no_email(self):
        request = rl.observe(self.store) | dict(id='run', owner=idle.OWNER, phase='complete',
                                               returncode=0, started_at=1, timeout_seconds=None)
        self.store.put(request)
        self.assertTrue(runtime.idle_exit_needs_email(self.store, request))
        with self.store.connect() as db:
            db.execute('INSERT INTO script_mail VALUES (?,?,?)', ('run', 'success', 1))
        self.assertFalse(runtime.idle_exit_needs_email(self.store, request))
        request.update(no_email=True, returncode=-9)
        self.assertFalse(runtime.idle_exit_needs_email(self.store, request))
        self.store.event(request, 'started', 'start')
        sender = Mock(return_value=True)
        runtime.deliver_timeout_events(sender, self.store)
        sender.assert_not_called()

    def test_fit_guard_nonblocking_and_released(self):
        with patch.object(FitLock, 'LOCK_DIR', self.root), patch.object(FitLockNN, 'LOCK_DIR', self.root):
            with rl.fit_guard():
                self.assertTrue(FitLock.is_held())
                with self.assertRaises(portalocker.LockException):
                    with rl.fit_guard():
                        pass
            self.assertFalse(FitLock.is_held())

    def test_execute_snapshot_and_result_failure(self):
        request = rl.observe(self.store) | dict(id='run', owner=idle.OWNER, phase='running')
        self.store.put(request)
        main = Mock(return_value=SimpleNamespace(success=True, execution_success=False))
        with patch.object(rl, 'fit_guard'), \
             patch('src.proj.util.cli.script_session.as_script_main'), \
             patch('src.proj.util.filesys.dynamic_import.dynamic_modules',
                   return_value=[SimpleNamespace(main=main)]):
            self.assertEqual(rl.execute(request, self.store), 1)
        path = Path(main.call_args.kwargs['config_path'])
        config = json.loads(path.read_text())
        self.assertTrue(config['output_root'].endswith('automatic/run'))
        self.assertEqual(config['scheduling_provenance']['version'], request['version'])
        self.assertEqual(evidence(self.store, 'run')['script_outcome'], 'error')
        self.assertTrue((path.parent / 'rl_experiment.yaml').is_file())

    def test_changed_version_and_lock_contention_defer_before_training(self):
        request = rl.observe(self.store) | dict(id='run', owner=idle.OWNER, phase='running')
        with patch.object(rl, 'fit_guard', side_effect=portalocker.LockException):
            self.assertEqual(rl.execute(request, self.store), 75)
        self.doc['rerun_mark'] = 'changed'
        self.commit()
        with patch.object(rl, 'fit_guard'), patch('src.proj.util.filesys.dynamic_import.dynamic_modules') as load:
            self.assertEqual(rl.execute(request, self.store), 75)
            load.assert_not_called()

    def test_running_version_finishes_before_latest_is_queued(self):
        request = rl.observe(self.store) | dict(id='run', owner=idle.OWNER, phase='running', started_at=1)
        self.store.put(request)
        self.doc['rerun_mark'] = 'newer'
        self.commit()
        idle.dispatch(self.options, self.store)
        self.assertEqual(len(self.store.owned(idle.OWNER)), 1)
        self.assertEqual(self.store.get('run')['version'], request['version'])
        request['phase'] = 'complete'
        self.store.put(request)
        idle.dispatch(self.options, self.store)
        self.assertNotEqual(self.store.owned(idle.OWNER)[-1]['version'], request['version'])

    def test_invalid_config_notification_deduplicated(self):
        self.doc['mode'] = 'invalid'
        self.commit()
        self.assertIsNone(rl.observe(self.store))
        self.assertIsNone(rl.observe(self.store))
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 1)
        self.assertEqual(self.store.owned(idle.OWNER), [])

    def test_invalid_config_notification_recipient_and_opt_out(self):
        self.doc.update(mode='invalid', recipient='research@example.invalid')
        self.commit()
        rl.observe(self.store)
        sender = Mock(return_value=True)
        runtime.deliver_timeout_events(sender, self.store)
        self.assertEqual(sender.call_args.kwargs['recipient'], 'research@example.invalid')
        self.doc['no_email'] = True
        self.commit()
        rl.observe(self.store)
        sender.reset_mock()
        runtime.deliver_timeout_events(sender, self.store)
        sender.assert_not_called()

    def test_shared_worker_installed_when_only_rl_is_enabled(self):
        from src.api.task_monitor.scheduling.config import load_config
        from src.api.task_monitor.scheduling.installer import build_plan, validate_unit
        # The committed installer documents, not the temporary Git fixture.
        root = Path(__file__).resolve().parents[4]
        config = load_config(root / 'runs/scheduling', 'mengkjin-server')
        config['watchdog']['jobs']['idle_worklist']['enabled'] = False
        plan = build_plan(config, root=self.root, python=Path(sys.executable), cron='', local_timezone='Asia/Shanghai')
        self.assertIn('learndl-idle-worklist.service', plan['units'])
        validate_unit('learndl-idle-worklist.service', plan['units']['learndl-idle-worklist.service'])

    def test_worker_rechecks_configuration_and_does_not_launch_twice(self):
        idle.dispatch(self.options, self.store)
        request = self.store.owned(idle.OWNER)[-1]
        with patch.object(idle, 'RunStore', return_value=self.store), \
             patch.object(idle, 'settings', return_value=self.options), \
             patch.object(idle, '_launch', return_value=0) as launch:
            self.assertEqual(idle.worker(), 0)
            self.assertEqual(idle.worker(), 0)
            launch.assert_called_once()
        self.assertEqual(self.store.get(request['id'])['phase'], 'starting')

    def test_cpu_script_pipeline_and_bundle_attachment_callback(self):
        """Actual PPO/report/bundle, synthetic input, isolated task and mail."""
        import inspect
        from src.proj import Proj
        from src.proj.util.cli.script_session import as_script_main
        from src.proj.util.filesys.dynamic_import import dynamic_modules
        from src.res.rl_experimental.data import make_synthetic_panel
        from src.res.rl_experimental import experiment
        script = PATH.scpt / '4_train' / '3_train_rl_experiment.py'
        snapshot = self.root / 'snapshot'
        snapshot.mkdir()
        for name in ('manifest.json', 'quality_report.json'):
            (snapshot / name).write_text('{}')
        self.doc['single']['training'].update(candidate_count=8, max_holdings=4, max_weight=.3,
                                              n_envs=1, rollout_steps=8, batch_size=8,
                                              update_epochs=1, hidden_dim=16, total_timesteps=16)
        self.commit()
        request = rl.observe(self.store) | dict(id='cpu-smoke', owner=idle.OWNER, phase='running')
        self.store.put(request)
        with as_script_main(script):
            module = next(iter(dynamic_modules(script)))
        original = inspect.unwrap(module.main)
        fake_task = SimpleNamespace(kwargs={}, email=True)

        def call(**kwargs):
            result = original(config_path=kwargs['config_path'], recipient=kwargs['recipient'],
                              no_email=kwargs['no_email'])
            return SimpleNamespace(success=result['status'] == 'success', execution_success=True)

        with patch.object(rl, 'fit_guard'), \
             patch('src.proj.util.filesys.dynamic_import.dynamic_modules', return_value=[SimpleNamespace(main=call)]), \
             patch.object(module, 'script_tool', SimpleNamespace(autorun_task=fake_task)), \
             patch.object(Proj, 'email_attachments', []), \
             patch.object(experiment, '_snapshot', return_value=(make_synthetic_panel(70, 24, 4, seed=5), snapshot)):
            self.assertEqual(rl.execute(request, self.store), 0)
            attachments = list(Proj.email_attachments)
        self.assertTrue(any(str(p).endswith('.json') for p in attachments))
        self.assertTrue(any(str(p).endswith('.zip') for p in attachments))
        result_root = self.root / 'results/automatic/cpu-smoke'
        run = next((result_root / 'runs').iterdir())
        metrics = json.loads((run / 'metrics.json').read_text())
        self.assertEqual(metrics['completed_updates'], 2)
        saved = json.loads((run / 'experiment_config.json').read_text())
        self.assertEqual(saved['scheduling_provenance']['version'], request['version'])
