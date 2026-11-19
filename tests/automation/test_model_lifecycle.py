"""Lifecycle repairs share idle-lane deduplication and bounded recovery."""
from __future__ import annotations

from unittest import TestCase
from unittest.mock import Mock, patch

from src.automation.scheduling.model_lifecycle import candidate, execute
from src.res.model.util.lifecycle import ModelLifecycle

__all__ = ['LifecycleQueueTest']


class LifecycleQueueTest(TestCase):
    def setUp(self):
        self.plan = {'name': 'gru@test', 'base': '/unused', 'version': 'v1', 'stage': 'production'}
        self.enterContext(patch.object(ModelLifecycle, 'inspect_registered', return_value=[self.plan]))
        self.options = {'enabled': True}
        self.store = Mock()
        self.store.owned.return_value = []

    def test_idle_request_has_durable_identity(self):
        first = candidate(self.store, self.options)
        self.assertEqual(first, candidate(self.store, self.options))
        assert first is not None
        self.assertEqual(first['task_kind'], 'model_lifecycle')
        self.assertEqual(first['task'], 'lifecycle:gru@test')
        self.assertEqual(first['lifecycle_request'], self.plan)

    def test_existing_request_is_not_duplicated(self):
        for phase in ('queued', 'starting', 'running', 'complete'):
            self.store.owned.return_value = [{'task': 'lifecycle:gru@test', 'version': 'v1', 'phase': phase}]
            self.assertIsNone(candidate(self.store, self.options))

    def test_failed_request_obeys_recovery_policy(self):
        self.store.owned.return_value = [{'task': 'lifecycle:gru@test', 'version': 'v1', 'phase': 'failed'}]
        with patch('src.automation.scheduling.recovery.recovery_plan', return_value={'retry': False}):
            self.assertIsNone(candidate(self.store, self.options))
        with patch('src.automation.scheduling.recovery.recovery_plan',
                   return_value={'retry': True, 'parent_run_id': 'old', 'retry_index': 1}):
            request = candidate(self.store, self.options)
            assert request is not None
            self.assertEqual(request['parent_run_id'], 'old')

    def test_failed_execution_records_error(self):
        with patch.object(ModelLifecycle, 'repair', side_effect=ValueError('failed')):
            with patch('src.automation.scheduling.recovery.record') as record:
                with self.assertRaises(ValueError):
                    execute({'id': 'job', 'lifecycle_request': self.plan}, self.store)
                self.assertEqual(record.call_args.kwargs['script_outcome'], 'error')
