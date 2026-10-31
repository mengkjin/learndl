import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.res.rl_experimental.progress import progress, quiet_output
from src.proj.util.web.task_mail import TaskResultMail, retry_delivery


class FakeEmail:
    @staticmethod
    def recipient(value):
        return value or 'test@example.com'

    @staticmethod
    def message(title, body, recipient, *, attachments):
        msg = MIMEMultipart()
        msg['To'], msg['Subject'] = recipient, title
        msg.attach(MIMEText(body))
        for path in attachments:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(Path(path).read_bytes())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment', filename=Path(path).name)
            msg.attach(part)
        return msg


class TaskMailTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('src.proj.env.MACHINE.emailable', True))

    def test_disabled_machine_skips_before_preparing_or_consuming_attachments(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('src.proj.env.MACHINE.emailable', False), \
             patch('src.proj.util.web.task_mail.prepare') as prepare, \
             patch('src.proj.util.web.emailer.Email.send_with_smtplib') as smtp, \
             patch('src.proj.Proj.email_attachments.pop_all') as pop:
            directory = Path(tmp)/'mail'
            api = Mock()
            self.assertTrue(TaskResultMail(directory, email_api=api)(
                'result', 'body', attachments=[Path(tmp)/'missing.html'], project_attachments=True))
            api.message.assert_not_called()
            prepare.assert_not_called()
            smtp.assert_not_called()
            pop.assert_not_called()
            custom = Mock()
            self.assertTrue(TaskResultMail(directory, transport=custom)('result', 'body'))
            custom.assert_not_called()
            self.assertFalse(directory.exists())

    def test_disabled_retry_preserves_old_failure_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ledger = directory/'delivery.json'
            original = b'{"status":"failed","messages":[{"sent":false,"attempts":1}]}'
            ledger.write_bytes(original)
            with patch('src.proj.env.MACHINE.emailable', False), \
                 patch('src.proj.util.web.emailer.Email.send_with_smtplib') as smtp:
                self.assertTrue(retry_delivery(directory))
                self.assertTrue(retry_delivery(directory/'not-created'))
                smtp.assert_not_called()
            self.assertEqual(ledger.read_bytes(), original)
            self.assertEqual(list(directory.iterdir()), [ledger])

    def test_enabled_default_transport_failure_is_still_reported(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('src.proj.env.MACHINE.emailable', True), \
             patch('src.proj.util.web.emailer.Email.send_with_smtplib', return_value=False) as smtp:
            self.assertFalse(TaskResultMail(tmp, email_api=FakeEmail)('result', 'body'))
            smtp.assert_called_once()
            record = json.loads((Path(tmp)/'delivery.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['messages'][0]['attempts'], 1)

    def test_autorun_uses_opt_in_sender_and_records_only_full_delivery(self):
        from src.proj.util.script.autorun import AutoRunTask
        from src.proj.util.web.emailer import Email
        task = SimpleNamespace(execution_status='Success', email=True, task_name='rl_experiment',
            time_str='now', source='script', task_full_name='rl_experiment', init_time=datetime.now(),
            end_time=datetime.now(), status='Success', error_message='', exit_message='',
            exit_files=['closed-task.html'], kwargs={'email_recipient': 'test@example.com'})
        with patch('src.automation.scheduling.runtime.termination_requested', return_value=False), \
             patch('src.automation.scheduling.recovery.script_result'), \
             patch('src.automation.scheduling.runtime.record_script_email') as receipt, \
             patch.object(Email, 'send', return_value=True) as default:
            AutoRunTask.send_email(task)
            default.assert_called_once()
            receipt.assert_called_once_with(success=True)
            receipt.reset_mock()
            task.email_sender = Mock(return_value=False)
            AutoRunTask.send_email(task)
            receipt.assert_not_called()
            self.assertEqual(task.email_sender.call_args.kwargs['attachments'], ['closed-task.html'])
            self.assertTrue(task.email_sender.call_args.kwargs['project_attachments'])
            task.email_sender.return_value = True
            AutoRunTask.send_email(task)
            receipt.assert_called_once_with(success=True)
            default.assert_called_once()  # unrelated task behavior unchanged

    def test_bounded_batches_retry_only_failed_and_keep_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = []
            for i in range(3):
                p = root/f'part{i}.zip'
                p.write_bytes(os.urandom(60000))
                files.append(p)
            html = root/'task.html'
            html.write_text('<p>training detail</p>'*40000)
            calls = []

            def deliver(message, recipient):
                self.assertEqual(recipient, 'chosen@example.com')
                self.assertLessEqual(len(message.as_bytes()), 150000)
                calls.append(message)
                return len(calls) != 2

            mail = TaskResultMail(root/'mail', max_bytes=150000, email_api=FakeEmail, transport=deliver)
            self.assertFalse(mail('RL result', 'Completed', 'chosen@example.com', attachments=[html, *files]))
            record = json.loads((root/'mail/delivery.json').read_text())
            self.assertEqual(len(record['messages']), 3)
            self.assertEqual([x['sent'] for x in record['messages']], [True, False, True])
            names = [p.get_filename() for m in calls for p in m.walk() if p.get_filename()]
            self.assertTrue(any(name.endswith('.zip') and 'part' not in name for name in names))
            self.assertTrue(retry_delivery(root/'mail', transport=deliver))
            self.assertEqual(len(calls), 4)
            record = json.loads((root/'mail/delivery.json').read_text())
            self.assertEqual([x['attempts'] for x in record['messages']], [1, 2, 1])
            self.assertTrue(retry_delivery(root/'mail', transport=deliver))
            self.assertEqual(len(calls), 4)

    def test_small_results_one_task_email_and_corruption_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root/name for name in ('task.html', 'results.zip', 'results.parts.json')]
            for p in files:
                p.write_text('small')
            calls = []
            mail = TaskResultMail(root/'mail', email_api=FakeEmail,
                                  transport=lambda msg, recipient: calls.append(msg) or False)
            self.assertFalse(mail('Result', 'body', attachments=files))
            self.assertEqual(len(calls), 1)
            self.assertEqual(len([p for p in calls[0].walk() if p.get_filename()]), 3)
            record = json.loads((root/'mail/delivery.json').read_text())
            (root/'mail'/record['messages'][0]['file']).write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                retry_delivery(root/'mail', transport=lambda *_: True)

    def test_in_flight_not_silently_resent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mail = TaskResultMail(root/'mail', email_api=FakeEmail, transport=lambda *_: False)
            mail('Result', 'body')
            path = root/'mail/delivery.json'
            state = json.loads(path.read_text())
            state['messages'][0]['in_flight'] = True
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(RuntimeError, 'Unknown prior SMTP'):
                retry_delivery(root/'mail', transport=lambda *_: self.fail('Must not resend'))

    def test_quiet_logs_preserve_details_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'detail.log'
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                with quiet_output(path):
                    progress('train', 'minibatch details')
                    progress('valuation', 'warning detail', warning=True)
                    progress('quarter', 'training complete', milestone=True)
                    progress('quarter', 'test complete', milestone=True)
            self.assertEqual(len(out.getvalue().splitlines()), 2)
            self.assertIn('minibatch details', path.read_text())
            self.assertIn('warning detail', path.read_text())
            with redirect_stdout(out), patch('src.res.rl_experimental.progress.time.monotonic', side_effect=[0, 3601, 3601]):
                with quiet_output(path):
                    print('actual progress')
            self.assertIn('heartbeat', out.getvalue())


if __name__ == '__main__':
    unittest.main()
