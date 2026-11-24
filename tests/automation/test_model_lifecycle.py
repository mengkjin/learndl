"""Old queued lifecycle jobs yield ownership to daily_update."""
from unittest import TestCase
from unittest.mock import Mock

from src.automation.scheduling.idle_worklist import _defer_retired_requests


class LifecycleQueueTest(TestCase):
    def test_only_queued_legacy_jobs_are_deferred(self):
        rows = [{'task_kind': 'model_lifecycle', 'phase': phase} for phase in ('queued', 'running', 'complete')]
        rows.append({'task_kind': 'model', 'phase': 'queued'})
        store = Mock()
        store.owned.return_value = rows
        _defer_retired_requests(store)
        store.put.assert_called_once_with(rows[0])
        self.assertEqual([r['phase'] for r in rows], ['deferred', 'running', 'complete', 'queued'])
        self.assertIn('daily_update', rows[0]['blocked_by'])
