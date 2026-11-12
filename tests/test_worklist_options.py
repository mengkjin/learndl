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
from src.api.calls.worklist_state import AutomaticDeferred, WorklistState
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
        self.worklist = self.root / 'worklist.yaml'
        self.schedule = self.root / 'schedule.yaml'
        self.schedule.write_text('model.module: gru\n')
        self.enterContext(patch.object(PATH, 'lc_machine', self.root))
        self.enterContext(patch.object(PATH, 'sched_worklist', self.worklist))
        self.enterContext(patch('src.res.model.util.training_history.git', return_value=None))
        self.enterContext(patch.object(ScheduleConfig, 'find_path', return_value=self.schedule))
        self.enterContext(patch.object(ScheduleConfig, 'check_name_exist', return_value=True))
        self.enterContext(patch('src.api.calls.research.MACHINE', SimpleNamespace(platform_server=True)))
        self.enterContext(patch('src.api.calls.research.as_script_main', side_effect=lambda *_: nullcontext()))
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
