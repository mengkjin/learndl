"""Tests for maintenance notice delivery and scheduling state."""
from __future__ import annotations

__all__ = [
    'NoticeTest',
]
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.automation.scheduling import maintenance_notice as notice
from src.automation.scheduling.auxiliary import control_notice
import json


class NoticeTest(unittest.TestCase):
    def test_snapshot_render_is_stable_and_identifies_operation(self):
        state = dict(owner='git', requested_at=0, phase='cooldown', cooldown_until=4500,
                     last_failure='gate.lock occupied')
        snapshot = dict(state=state, captured_at=3600, host='host',
                        tasks=[dict(task='training', id='r', pid=42, channel='research', started_at=0,
                                    status='alive', log='/tmp/train.log')], limitations=[])
        title, body = control_notice('RuntimeError: ' + notice.PREFIX + json.dumps(snapshot))
        self.assertIn('Git update', title)
        self.assertIn('PID 42', body)
        self.assertIn('1h 0m', body)
        self.assertIn('not proof of lock ownership', body)
        self.assertIn('1970-01-01', body)
        state['owner'] = 'installation'
        self.assertIn('Watchdog installation', notice.render(snapshot)[0])
        state['owner'] = 'autofix:repair'
        self.assertIn('repair', notice.render(snapshot)[0])

    def test_missing_ledgers_do_not_get_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(notice, 'runtime_dir', return_value=root), patch.object(notice, 'state_dir', return_value=root):
                snapshot = notice.capture(dict(owner='git', requested_at=0))
            self.assertFalse(list(root.iterdir()))
            self.assertEqual(len(snapshot['limitations']), 2)
            self.assertIn('No matching managed task', notice.render(snapshot)[1])
