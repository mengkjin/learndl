"""Exercise host-local scheduling with real process identity and file locks."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from unittest import TestCase
from unittest.mock import patch

from src.automation.scheduling import idle_worklist, runtime, training_maintenance, training_scheduler as scheduler
from src.shared.file_ops import atomic_json


CLIENT = """
import sys
import time
from pathlib import Path
from src.automation.scheduling import training_scheduler as scheduler

root, name, kind, resource = sys.argv[1:]
root = Path(root)
scheduler.runtime_dir = lambda: root
scheduler.available = lambda: True
scheduler._fits_occupied = lambda: False

try:
    with scheduler.admit(kind, name, request_id=name, resource=resource):
        (root / f'{name}.active').touch()
        while not (root / f'{name}.release').exists():
            if scheduler.pause_request():
                (root / f'{name}.paused').touch()
                break
            time.sleep(.06)
except KeyboardInterrupt:
    (root / f'{name}.cancelled').touch()
"""


class SchedulerTest(TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(scheduler, 'available', return_value=True))
        self.enterContext(patch.object(scheduler, '_fits_occupied', return_value=False))
        for module in (scheduler, runtime, training_maintenance, idle_worklist):
            self.enterContext(patch.object(module, 'runtime_dir', return_value=self.root))

    @contextmanager
    def process(self, name, kind, resource='all'):
        process = subprocess.Popen([sys.executable, '-c', CLIENT, str(self.root), name, kind, resource],
                                   stdout=subprocess.DEVNULL)
        try:
            yield process
        finally:
            (self.root / f'{name}.release').touch()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise

    def until(self, predicate, message='scheduler state was not reached'):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(.03)
        self.fail(message)

    def marker(self, name):
        return self.until((self.root / name).exists, name)

    def request(self, name, phase):
        def row():
            current = scheduler.status()['requests'].get(name)
            return current if current and current['phase'] == phase else None
        return self.until(row, f'{name} never reached {phase}')

    def test_adjacent_priorities_yield_only_after_checkpoint(self):
        scheduler.status('unified')
        for low, high in (('idle', 'promotion'), ('promotion', 'manual'), ('manual', 'production')):
            with self.subTest(low=low, high=high):
                lower, higher = f'{low}-fit', f'{high}-request'
                with self.process(lower, low) as left:
                    self.marker(f'{lower}.active')
                    with self.process(higher, high):
                        self.marker(f'{lower}.paused')
                        self.marker(f'{higher}.active')
                        left.wait(timeout=5)
                        self.assertEqual(scheduler.status()['mode'], 'unified')

    def test_gpu_fifo_cpu_concurrency_and_foreground_cancel(self):
        scheduler.status('unified')
        with self.process('first', 'manual', 'nn'):
            self.marker('first.active')
            with self.process('cpu', 'manual', 'shared'):
                self.marker('cpu.active')
            with self.process('second', 'manual', 'nn') as second:
                self.request('second', 'waiting')
                with self.process('third', 'manual', 'nn'):
                    self.request('third', 'waiting')
                    second.send_signal(signal.SIGINT)
                    self.marker('second.cancelled')
                    second.wait(timeout=5)
                    self.assertNotIn('second', scheduler.status()['requests'])
                    (self.root / 'first.release').touch()
                    self.marker('third.active')

    def test_switch_naturally_then_recover_dead_owner(self):
        self.assertEqual(scheduler.status()['mode'], 'legacy')
        with self.process('old', 'idle') as old:
            self.marker('old.active')
            state = scheduler.status('unified')
            self.assertEqual((state['mode'], state['target'], state['transition_blockers']),
                             ('legacy', 'unified', ['old']))
            with self.process('new', 'manual') as new:
                row = self.request('new', 'waiting')
                self.assertIn('transition', row['blocked_by'])
                self.assertFalse((self.root / 'old.paused').exists())
                (self.root / 'old.release').touch()
                self.marker('new.active')
                old.wait(timeout=5)
                state = scheduler.status('legacy')
                self.assertEqual((state['mode'], state['target']), ('unified', 'legacy'))
                new.kill()
                new.wait(timeout=5)
                self.assertEqual(scheduler.status()['mode'], 'legacy')

    def test_observe_records_priority_without_pausing_and_readiness_expires(self):
        scheduler.status('observe')
        with self.process('idle', 'idle'):
            self.marker('idle.active')
            demand = {'task': 'promotion:model', 'version': 'upgrade'}
            scheduler.discover_promotion(demand, False)
            self.assertNotIn('promotion', scheduler.status())
            scheduler.discover_promotion(demand, True)
            events = self.until(lambda: [event for event in scheduler.status()['observations']
                                         if event['point'] == 'checkpoint' and event['decision'] == 'upgrade'])
            self.assertEqual(events[-1]['kind'], 'idle')
            self.assertFalse((self.root / 'idle.paused').exists())
            scheduler.discover_promotion(None, False)
            self.assertNotIn('promotion', scheduler.status())
            with self.process('manual', 'manual'):
                self.marker('manual.active')

    def test_production_admission_stays_active_across_batch_gaps(self):
        scheduler.status('unified')
        with self.process('production', 'production'):
            self.marker('production.active')
            with self.process('research', 'idle'):
                self.request('research', 'waiting')
                self.assertFalse((self.root / 'research.active').exists())
                (self.root / 'production.release').touch()
                self.marker('research.active')

    def test_paused_promotion_keeps_candidate_on_rollback_and_respects_cancellation(self):
        from src.automation.scheduling import model_lifecycle
        from src.res.model.util import lifecycle_promotion
        from src.res.model.workflow.worklist_state import AutomaticDeferred

        scheduler.status('unified')
        checkpoint = self.root / 'candidate'
        checkpoint.mkdir()
        request = {'base': str(self.root / 'source'), 'name': 'gru@source', 'version': 'promotion-version'}
        ticket = {'id': 'a' * 32, 'entry': 'promotion', 'managed_parent': 'parent', 'ready': True,
                  'worklist': None, 'model_path': str(checkpoint), 'weights': {}, 'config_files': {},
                  'email': False, 'schedule_name': None, 'training_priority': 'promotion',
                  'training_requested_at': 10, 'promotion_request': request}
        atomic_json(self.root / 'maintenance-resumes' / f'{ticket["id"]}.json', ticket)
        store = runtime.RunStore()
        store.put({'id': 'parent', 'owner': 'idle_worklist', 'phase': 'deferred', 'returncode': 75})

        self.assertEqual(scheduler.status('legacy')['mode'], 'legacy')
        waiting = training_maintenance.candidate(store)
        self.assertIsNotNone(waiting)
        assert waiting is not None
        self.assertEqual(waiting['training_requested_at'], 10)
        with (patch.object(lifecycle_promotion, 'still_requested', return_value=True),
              patch.object(model_lifecycle, 'execute', return_value=0) as resume):
            self.assertEqual(training_maintenance.execute(waiting), 0)
        self.assertEqual(resume.call_args.args[0]['promotion_request'], request)
        self.assertEqual(resume.call_args.kwargs['expected_candidate'], checkpoint)

        with patch.object(lifecycle_promotion, 'still_requested', return_value=False):
            with self.assertRaises(AutomaticDeferred):
                training_maintenance.execute(waiting)
        training_maintenance.cancel(store, ticket['id'])
        self.assertIsNone(training_maintenance.candidate(store))
