"""Exercise maintenance state and fit boundaries with a temporary scheduler ledger."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import os
import time

from src.automation.scheduling import idle_worklist, training_maintenance as maintenance
from src.automation.scheduling.runtime import RunStore, idle_exit_needs_email
from src.proj.util.script.fit_lock import FitLock, FitLockNN


class TrainingMaintenanceTest(TestCase):
    def setUp(self):
        temp = self.enterContext(TemporaryDirectory())
        self.root = Path(temp)
        self.store = RunStore(self.root / 'runs.sqlite')
        self.enterContext(patch.object(maintenance, 'runtime_dir', return_value=self.root))
        self.enterContext(patch.object(idle_worklist, 'runtime_dir', return_value=self.root))
        self.enterContext(patch.object(maintenance, 'installed_config', return_value={}))
        self.enterContext(patch.object(FitLock, 'LOCK_DIR', self.root))
        self.installation = self.enterContext(patch(
            'src.automation.scheduling.install_request.current', return_value={}))
        self.enterContext(patch.dict(os.environ, {'LEARNDL_MANAGED_RUN': '', 'LEARNDL_AUTOFIX_ID': ''}))

    def test_verified_request_and_scoped_authorized_install(self):
        self.assertIsNone(maintenance.maintenance_request('research'))
        maintenance.publish_request('git', 'new-head')
        self.assertEqual(maintenance.maintenance_request('research')['id'], 'new-head')
        maintenance.publish_request('git')
        self.installation.return_value = {
            'id': 'install-1', 'phase': 'waiting', 'groups': ['manual'],
            'boot_id': 'boot', 'expires_at': time.time() + 60,
            'admission_plan': {'impact': {'mode': 'exclusive'}},
        }
        with patch('src.automation.scheduling.runtime.boot_id', return_value='boot'):
            self.assertIsNone(maintenance.maintenance_request('research'))
            self.assertEqual(maintenance.maintenance_request('manual')['id'], 'install-1')
            self.installation.return_value['phase'] = 'cancelled'
            self.assertIsNone(maintenance.maintenance_request('manual'))

    def test_checkpoint_prepares_then_enqueues_and_reopens_exact_directory(self):
        base = self.root / 'model'
        for number in (0, 1):
            path = base / f'archive/{number}/20240102/best/state_dict.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'saved weights')
        conf = base / 'configs'
        conf.mkdir()
        (conf / 'model.yaml').write_text('model: saved\n')
        config = SimpleNamespace(
            module_type='nn', is_null_model=False, base_path=SimpleNamespace(base=base, conf=lambda: conf),
            submodels=['best'], boost_head_config=None, model_num_list=[0, 1], short_test=False,
            beg_date=20240102, end_date=20240103, queue_of_stages=['fit', 'test'],
            model_config=SimpleNamespace(schedule_name=None),
        )
        trainer = SimpleNamespace(config=config, model_date=20240102, model_num=1,
                                  data=SimpleNamespace(model_date_list=[20240102, 20240103]),
                                  training_run=SimpleNamespace(run_id='a' * 32), logger=Mock())
        maintenance.publish_request('git', 'new-head')
        with (patch.object(maintenance, 'available', return_value=True),
              patch('src.proj.Save.async_wait_all') as sync):
            @maintenance.checkpointed_script('model')
            def train():
                try:
                    maintenance.checkpoint(trainer)
                finally:
                    self.assertIsNone(maintenance.candidate(self.store))
            pause = train()
        sync.assert_called_once()
        self.assertEqual(pause.ticket['completed'], [[20240102, 0], [20240102, 1]])
        request = maintenance.candidate(self.store)
        self.assertEqual(request['id'], pause.ticket['id'])
        with maintenance.continuation(pause.ticket):
            args, kwargs = {}, {}
            maintenance.configure_resume(args, kwargs)
            self.assertEqual(args['base_path'], base)
            self.assertEqual((kwargs['resume'], kwargs['stage'], kwargs['end']), (1, 0, 20240103))
            maintenance.validate_remaining_nodes([(20240102, 0), (20240102, 1), (20240103, 0)], [(20240103, 0)])
            (base / 'archive/0/20240102/best/state_dict.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checkpoint is missing or changed'):
                maintenance.validate_active_checkpoint()
        idle_worklist._queue(request, self.store, {})
        self.assertIsNone(maintenance.candidate(self.store))
        maintenance.cancel(self.store, request['id'])
        self.assertEqual(self.store.get(request['id'])['phase'], 'cancelled')
        self.assertIsNone(maintenance.candidate(self.store))

    def test_daily_waits_for_active_fit_and_takes_priority_over_queued_nn_fit(self):
        production_started, production_done, research_started, research_queued = (Event() for _ in range(4))
        nn_guard = FitLockNN.guard

        def queued_nn_guard(**kwargs):
            research_queued.set()
            return nn_guard(**kwargs)

        def production():
            with FitLock.production_guard(), FitLock.guard():
                production_started.set()
                self.assertTrue(production_done.wait(3))

        def research():
            with FitLock.trainer_guard(try_cuda=True, respect_production=True):
                research_started.set()

        with (ThreadPoolExecutor(max_workers=2) as pool,
              patch.object(FitLockNN, 'LOCK_DIR', self.root),
              patch.object(FitLockNN, 'enabled', return_value=True),
              patch.object(FitLockNN, 'guard', side_effect=queued_nn_guard)):
            try:
                with FitLock.guard(), nn_guard():
                    second = pool.submit(research)
                    self.assertTrue(research_queued.wait(3))
                    first = pool.submit(production)
                    deadline = time.monotonic() + 3
                    while not FitLock.production_requested() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertEqual(maintenance.maintenance_request('research')['kind'], 'daily_update')
                    self.assertEqual(maintenance.maintenance_request('manual')['kind'], 'daily_update')
                    self.assertFalse(production_started.is_set())
                self.assertTrue(production_started.wait(3))
                self.assertFalse(research_started.is_set())
            finally:
                production_done.set()
            first.result(timeout=3)
            second.result(timeout=3)
        self.assertTrue(research_started.is_set())
        self.assertIsNone(maintenance.maintenance_request('research'))

    def test_managed_pause_waits_for_exit_and_respects_stop(self):
        ticket = {'id': 'b' * 32, 'entry': 'model', 'managed_parent': 'original',
                  'model_path': '/saved/model', 'worklist': None, 'schedule_name': None,
                  'email': True, 'maintenance': {'kind': 'patch'}}
        maintenance.enqueue(maintenance.TrainingPaused(ticket))
        parent = {'id': 'original', 'phase': 'running', 'owner': idle_worklist.OWNER}
        self.store.put(parent)
        self.assertIsNone(maintenance.candidate(self.store))
        self.store.put(dict(parent, phase='error', returncode=-15))
        self.assertIsNone(maintenance.candidate(self.store))
        self.store.put(dict(parent, phase='deferred', returncode=75, suppress_auto_recovery=True))
        self.assertIsNone(maintenance.candidate(self.store))
        self.store.put(dict(parent, phase='error'))  # Parent could not persist its return code before reboot.
        maintenance.record_pause(self.store, parent['id'])
        self.assertFalse(idle_exit_needs_email(self.store, self.store.get(parent['id'])))
        self.assertEqual(maintenance.candidate(self.store)['id'], ticket['id'])

    def test_script_cleanup_does_not_deliver_pause_as_a_result(self):
        from src.proj.util.script.autorun import AutoRunTask

        task = AutoRunTask('train_model')
        task.catchers = Mock()
        pause = maintenance.TrainingPaused({'maintenance': {'kind': 'git'}})
        with (patch.object(task, 'send_email') as email, patch.object(task, '_finish_delivery') as delivery,
              patch('src.proj.Proj.exit_files.pop_all'), patch('src.proj.Proj.email_attachments.pop_all')):
            task._finish_task(type(pause), pause, None)
        email.assert_not_called()
        delivery.assert_not_called()
        self.assertEqual(task.status, 'Paused')

    def test_fit_checks_after_model_and_date_hooks_and_completes_empty_resume(self):
        from src.res.model.util.trainer.base_trainer import BaseTrainer

        events = []
        trainer = Mock()
        trainer.config.model_num_list = [0, 1]
        trainer.iter_model_num_date.return_value = [(20240102, 0), (20240102, 1)]
        trainer.logger.paragraph.return_value = nullcontext()
        trainer.data.datas.date = [20240101]
        trainer.data.d0, trainer.data.d1 = 0, 1
        trainer.on_fit_model_end.side_effect = lambda: events.append('dumped')
        trainer.on_fit_model_date_end.side_effect = lambda: events.append('date closed')
        trainer.on_fit_end_after.side_effect = lambda: events.append('fit closed')

        def pause_on_last_node(current):
            events.append('checked')
            if current.status.model_num == 1:
                raise maintenance.TrainingPaused({'maintenance': {'kind': 'git'}})

        with (patch.object(FitLock, 'trainer_guard', return_value=nullcontext()),
              patch('src.res.model.util.model_properties.track_phase', return_value=nullcontext()),
              patch('src.res.model.util.lifecycle.ModelLifecycle.record_node', side_effect=lambda _: events.append('recorded')),
              patch.object(maintenance, 'checkpoint', side_effect=pause_on_last_node)):
            with self.assertRaises(maintenance.TrainingPaused):
                BaseTrainer.stage_fit(trainer)
            self.assertEqual(events, ['dumped', 'recorded', 'checked', 'dumped', 'recorded', 'date closed', 'checked'])
            trainer.iter_model_num_date.return_value = []
            BaseTrainer.stage_fit(trainer)
        self.assertEqual(events[-1], 'fit closed')
        self.assertEqual(events.count('fit closed'), 1)

    def test_worker_waits_for_maintenance_without_consuming_queued_resume(self):
        idle_worklist._queue({'id': 'resume', 'task': 'manual', 'version': 'resume',
                              'task_kind': maintenance.TASK}, self.store, {})
        maintenance.publish_request('git', 'new-head')
        with (patch.object(idle_worklist, 'runtime_dir', return_value=self.root),
              patch.object(idle_worklist, 'RunStore', return_value=self.store),
              patch.object(idle_worklist.version_patch, 'business_lease', return_value=nullcontext()),
              patch('src.proj.util.script.script_lock.ScriptLock', return_value=nullcontext()),
              patch.object(maintenance, 'available', return_value=True),
              patch.object(idle_worklist, '_launch') as launch):
            self.assertEqual(idle_worklist.worker(), 0)
        launch.assert_not_called()
        self.assertEqual(self.store.get('resume')['phase'], 'queued')
