"""Outbox retries with simulated SMTP and real cross-process locking."""
import json
import smtplib
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.proj.util.web import emailer

Email = emailer.Email
ROOT = Path(__file__).resolve().parents[1]


class OutboxRetryTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(emailer, 'PATH', SimpleNamespace(email_outbox=self.root)))
        self.stack.enter_context(patch.object(emailer, 'MACHINE', SimpleNamespace(emailable=True)))
        self.stack.enter_context(patch.object(Email, 'logger', Mock()))
        self.deliver = self.stack.enter_context(patch.object(Email, '_deliver'))

    def enqueue(self, subject='result'):
        msg = EmailMessage()
        msg['To'] = 'nobody@example.invalid'
        msg['Subject'] = subject
        msg.set_content('simulated mail')
        return Email._enqueue(msg, '', 'simulated connection error')

    def test_empty_and_unpublished_entries_are_ignored(self):
        (self.root / '.pending.tmp').mkdir()
        (self.root / 'dead').mkdir()
        self.assertEqual(Email.retry_pending()['attempted'], 0)
        self.deliver.assert_not_called()

    def test_fifo_budget_and_recovery(self):
        for i in range(3):
            self.enqueue(str(i))
        self.deliver.side_effect = ConnectionError('offline')
        stats = Email.retry_pending()
        self.assertEqual((stats['attempted'], stats['sent'], stats['remaining']), (2,0,3))
        self.assertIn('offline', stats['error'])
        self.deliver.reset_mock(side_effect=True)
        with patch('src.proj.util.web.mail_queue.time.time', return_value=time.time()+61):
            stats = Email.retry_pending(max_messages=2)
        self.assertEqual((stats['sent'], stats['remaining']), (2,1))
        self.assertEqual([str(c.args[0]['Subject']) for c in self.deliver.call_args_list], ['0', '1'])
        self.assertEqual(Email.retry_pending()['sent'], 1)

    def test_deadline_checked_between_messages(self):
        self.enqueue('first')
        self.enqueue('second')
        clock = [100.0]
        self.deliver.side_effect = lambda *a, **kw: clock.__setitem__(0, 125.0)
        with patch.object(emailer.time, 'monotonic', side_effect=lambda: clock[0]):
            stats = Email.retry_pending(max_seconds=20)
        self.assertEqual((stats['sent'], stats['remaining']), (1,1))

    def test_rejections_reach_dead_letter_and_corrupt_entries_are_isolated(self):
        folder = self.enqueue()
        self.deliver.side_effect = smtplib.SMTPRecipientsRefused({'nobody': (550, b'rejected')})
        Email.retry_pending()  # Permanent recipient refusal isolates immediately.
        self.assertFalse(folder.exists())
        self.assertTrue((self.root / 'dead' / folder.name).exists())
        broken = self.root / 'broken'
        broken.mkdir()
        Email.retry_pending()
        self.assertTrue((self.root / 'dead' / 'broken').exists())

    def test_business_send_drains_retry_queue_without_duplicate(self):
        self.enqueue('old')
        msg = EmailMessage()
        msg['To'] = 'nobody@example.invalid'
        msg['Subject'] = 'new'
        with patch.object(Email, 'message', return_value=msg):
            self.assertTrue(Email.send('new'))
        self.assertEqual(Email.retry_pending()['sent'], 0)
        self.assertEqual([str(c.args[0]['Subject']) for c in self.deliver.call_args_list], ['old', 'new'])

    def _competing_process(self, mode):
        self.enqueue('queued')
        code = r'''
import json, sys, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from email.message import EmailMessage
from src.proj.util.web import emailer
root = Path(sys.argv[1])
def deliver(*args, **kwargs):
    (root / '.ready').touch()
    deadline = time.monotonic() + 30
    while not (root / '.release').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('parent did not release test sender')
        time.sleep(.02)
msg = EmailMessage()
msg['To'] = 'nobody@example.invalid'
msg['Subject'] = 'new'
with patch.object(emailer, 'PATH', SimpleNamespace(email_outbox=root)), patch.object(emailer, 'MACHINE', SimpleNamespace(emailable=True)), patch.object(emailer.Email, 'logger', Mock()), patch.object(emailer.Email, '_deliver', side_effect=deliver) as send, patch.object(emailer.Email, 'message', return_value=msg):
    result = emailer.Email.retry_pending() if sys.argv[2] == 'retry' else emailer.Email.send('new')
    print(json.dumps({'calls': send.call_count, 'result': result}))
'''
        child = subprocess.Popen([sys.executable, '-c', code, str(self.root), mode],
                                 cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 30
            while not (self.root / '.ready').exists() and child.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue((self.root / '.ready').exists(), 'child did not reach simulated SMTP')
            stats = Email.retry_pending()
            self.assertTrue(stats['lock_busy'])
            self.deliver.assert_not_called()
            self.assertEqual(len(Email._pending_outbox()), 1 if mode == 'retry' else 2)
            (self.root / '.release').touch()
            output, error = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, error)
            result = json.loads(output)
            self.assertEqual(result['calls'], 1 if mode == 'retry' else 2)
            self.assertEqual(Email.retry_pending()['sent'], 0)
            self.enqueue('after release')
            self.assertEqual(Email.retry_pending()['sent'], 1)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()

    def test_other_business_process_sending_is_not_preempted(self):
        self._competing_process('send')

    def test_two_retry_processes_do_not_deliver_same_message(self):
        self._competing_process('retry')


if __name__ == '__main__':
    unittest.main()
