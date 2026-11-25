"""Mixed worklist entries share validated options across dispatch and execution."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch
import tempfile

import yaml

from src.proj import PATH
from src.api.calls.research import CarryOutScheduleWorkList
from src.res.model.workflow.worklist_state import AutomaticDeferred, WorklistState
from src.automation.scheduling import idle_worklist as idle
from src.res.model.util.config.config import ScheduleConfig


class WorklistOptionsTest(TestCase):
    def test_mixed_entries_inherit_and_override(self):
        entries = WorklistState.entries({
            'resume': True, 'mode': 'quick', 'lifecycle': {'step': 2, 'window': 1000},
            'fit': ['one', {'name': 'two', 'resume': False, 'mode': 'sparse',
                            'lifecycle': {'step': 3}, 'start': 20200102}],
        })
        self.assertEqual(list(entries), ['one', 'two'])
        self.assertEqual(entries['one']['lifecycle'], {'mode': 'quick', 'step': 2, 'window': 1000})
        self.assertTrue(entries['one']['resume'])
        self.assertFalse(entries['two']['resume'])
        self.assertEqual(entries['two']['lifecycle'], {'mode': 'sparse', 'step': 3, 'window': 1000})

    def test_strings_keep_existing_defaults(self):
        self.assertEqual(WorklistState.entries({'fit': ['a', 'a']}), {'a': {'resume': False, 'force': False}})
        self.assertEqual(WorklistState.options({'fit': [], 'resume': True}, 'standalone'),
                         {'resume': True, 'force': False})

    def test_invalid_entries_rejected(self):
        for entry in ({'name': 'a', 'resume': 'false'}, {'name': 'a', 'mode': 'quik'},
                      {'name': 'a', 'resuem': True}, {'name': 'a', 'lifecycle': {'step': 0}},
                      {'name': 'a', 'lifecycle': {'step': True}}, {'name': 'a', 'lifecycle': {'stage': 'production'}},
                      {'name': 'a', 'start': 20260230}, {'name': 'a', 'start': 20260101, 'end': 20250101},
                      {'name': 'a', 'mode': 'quick', 'lifecycle': {'mode': 'sparse'}}, {'resume': True}):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                WorklistState.entries({'fit': [entry]})
        with self.assertRaisesRegex(ValueError, 'Conflicting worklist entries'):
            WorklistState.entries({'fit': ['a', {'name': 'a', 'mode': 'sparse'}]})


class WorklistExecutionTest(TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.worklist = self.root / 'worklists' / 'main.yaml'
        self.worklist.parent.mkdir()
        self.schedule = self.root / 'schedule.yaml'
        self.schedule.write_text('model.module: gru\n')
        self.enterContext(patch.object(PATH, 'lc_machine', self.root))
        self.enterContext(patch.object(PATH, 'sched_worklists', self.worklist.parent))
        self.enterContext(patch('src.res.model.workflow.worklists.git_time', return_value=1))
        self.enterContext(patch('src.res.model.util.training_history.git', return_value=None))
        self.enterContext(patch.object(ScheduleConfig, 'find_path', return_value=self.schedule))
        self.enterContext(patch.object(ScheduleConfig, 'check_name_exist', return_value=True))
        self.enterContext(patch('src.res.model.workflow.execution.MACHINE', SimpleNamespace(platform_server=True)))
        self.enterContext(patch('src.res.model.workflow.execution.as_script_main', side_effect=lambda *_: nullcontext()))
        self.enterContext(patch.object(CarryOutScheduleWorkList, '_release_gpu_memory'))
        self.enterContext(patch('src.res.model.util.training_history.collect_training_runs', self.collect))
        self.main = Mock(side_effect=self.train)
        self.enterContext(patch.object(CarryOutScheduleWorkList, '_load_schedule_main', return_value=self.main))
        self.runs = []
        self.config = {'resume': True, 'fit': ['one', {'name': 'two', 'resume': False, 'mode': 'sparse',
                                                      'lifecycle': {'step': 3}}]}
        self.write()

    def write(self):
        self.worklist.write_text(yaml.safe_dump(self.config))

    @contextmanager
    def collect(self, **kwargs):
        self.runs = []
        yield self.runs

    def train(self, **kwargs):
        self.runs.append({'run_id': kwargs['schedule_name'], 'status': 'success', 'model_path': str(self.root / 'model')})
        return SimpleNamespace(success=True, execution_success=True)

    def test_manual_worklist_forwards_each_entry(self):
        CarryOutScheduleWorkList().run()
        first, second = self.main.call_args_list
        self.assertTrue(first.kwargs['resume'])
        self.assertNotIn('lifecycle_options', first.kwargs)
        self.assertFalse(second.kwargs['resume'])
        self.assertEqual(second.kwargs['lifecycle_options'], {'mode': 'sparse', 'step': 3})
        state = WorklistState('two').read()
        assert state is not None
        self.assertEqual(state['options']['lifecycle'], {'mode': 'sparse', 'step': 3})
        CarryOutScheduleWorkList().run()
        self.assertEqual(self.main.call_count, 2)

    def test_dispatch_and_automatic_execution_use_same_item(self):
        store = Mock()
        store.owned.return_value = []
        WorklistState('one').save(status='success', revisions=WorklistState.revisions('one'))
        worklist, schedules, missing = idle.worklist_schedules(store)
        self.assertEqual(missing, [])
        with patch.object(WorklistState, 'decision', wraps=WorklistState.decision) as decide:
            request = idle._candidate(store, worklist, schedules, options={})
        assert request is not None
        self.assertEqual(request['task'], 'two')
        self.assertFalse(decide.call_args.kwargs['resume'])
        CarryOutScheduleWorkList(automatic=True, only_schedule='two', expected_version=request['version']).run()
        self.assertEqual(self.main.call_args.kwargs['lifecycle_options'], {'mode': 'sparse', 'step': 3})
        self.assertFalse(self.main.call_args.kwargs['resume'])
        self.assertEqual(self.main.call_args.kwargs['resume_selection'], 'latest')
        self.config['fit'][1]['mode'] = 'quick'
        self.write()
        with self.assertRaises(AutomaticDeferred):
            CarryOutScheduleWorkList(automatic=True, only_schedule='two', expected_version=request['version']).run()
        self.assertEqual(self.main.call_count, 1)

    def test_dictionary_membership_allows_explicit_request(self):
        WorklistState('two').request_training('request-1')
        self.assertTrue(WorklistState('two').request_path.is_file())

    def test_manual_batch_continues_active_item_then_remaining_items(self):
        from src.automation.scheduling import training_maintenance as maintenance

        self.enterContext(patch('src.res.model.workflow.worklists.git_time', return_value=None))
        def pause_first(**kwargs):
            paused = maintenance._worklist.get()
            raise maintenance.TrainingPaused({
                'id': 'test-run', 'maintenance': {'kind': 'git'}, 'worklist': paused,
            })

        self.main.side_effect = pause_first
        with patch.object(maintenance, 'enqueue') as enqueue:
            paused = CarryOutScheduleWorkList().run()
        enqueue.assert_called_once_with(paused)
        self.assertEqual(self.main.call_count, 1)
        self.assertEqual((WorklistState('one').read() or {}).get('status'), 'paused')
        self.assertEqual(paused.ticket['worklist']['remaining'], ['two'])
        self.assertFalse(paused.ticket['worklist']['automatic_source'])

        self.main.side_effect = self.train
        (self.worklist.parent / 'extra.yaml').write_text('fit: [three]\n')
        CarryOutScheduleWorkList(automatic=True).run()
        self.assertEqual(self.main.call_count, 1)
        CarryOutScheduleWorkList(automatic=True, maintenance_ticket=paused.ticket).run()
        self.assertEqual([call.kwargs['schedule_name'] for call in self.main.call_args_list], ['one', 'one', 'two'])
        self.assertTrue(self.main.call_args_list[1].kwargs['resume'])
        self.assertEqual(self.main.call_args_list[1].kwargs['resume_selection'], 'latest')
        for name in ('one', 'two'):
            self.assertEqual((WorklistState(name).read() or {}).get('status'), 'success')

    def test_legacy_manual_batch_keeps_order_and_skips_completed_items(self):
        extra = self.worklist.parent / 'remaining.yaml'
        extra.write_text('fit: [three, four, new]\n')
        self.enterContext(patch('src.res.model.workflow.worklists.git_time',
                                side_effect=lambda path: None if path == extra else 1))
        with patch('src.res.model.util.training_history.git', return_value='tracked'):
            revisions = WorklistState.revisions('one')
            WorklistState('one').save(status='paused', revisions=revisions, maintenance_resume_id='pause')
            WorklistState('two').save(status='success', revisions=WorklistState.revisions('two'))
            ticket = {'id': 'pause', 'managed_parent': 'previous-worker',
                      'worklist': {'name': 'one', 'remaining': ['four', 'two', 'three'], 'revisions': revisions}}
            CarryOutScheduleWorkList(automatic=True, maintenance_ticket=ticket).run()
        self.assertEqual([call.kwargs['schedule_name'] for call in self.main.call_args_list], ['one', 'four', 'three'])

    def test_superseded_paused_request_does_not_open_old_model(self):
        ticket = {'id': 'test-run', 'worklist': {'name': 'one', 'remaining': [], 'revisions': WorklistState.revisions('one')}}
        self.config['fit'] = [{'name': 'one', 'resume': False}]
        self.write()
        with self.assertRaisesRegex(AutomaticDeferred, 'superseded'):
            CarryOutScheduleWorkList(automatic=True, maintenance_ticket=ticket).run()
        self.main.assert_not_called()
