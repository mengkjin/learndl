"""Local outbox keeps email content when SMTP cannot be reached."""
from __future__ import annotations

import json
import smtplib
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch
from contextlib import contextmanager

from src.proj import MACHINE, PATH
from src.proj.util.web.emailer import Email


def parts(message: Message) -> tuple[str, list[bytes]]:
    texts: list[str] = []
    files: list[bytes] = []
    for part in message.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        filename = part.get_filename()
        if filename:
            files.append(payload)
        else:
            texts.append(payload.decode('utf-8'))
    return '\n'.join(texts), files


class EmailOutboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / 'email_outbox'
        self._outbox = patch.object(PATH, 'email_outbox', self.root)
        self._emailable = patch.object(MACHINE, 'emailable', True)
        self._outbox.start()
        self._emailable.start()

    def tearDown(self) -> None:
        self._emailable.stop()
        self._outbox.stop()
        self._tmp.cleanup()

    def test_failed_send_keeps_body_and_attachment_bytes(self) -> None:
        attachment = Path(self._tmp.name) / 'scores.csv'
        attachment.write_bytes(b'alpha,1\n')
        with patch.object(Email, '_deliver', side_effect=TimeoutError('timed out')):
            sent = Email.send(
                'Daily', 'body text', 'a@b.com',
                attachments=attachment, confirmation_message='Autorun',
            )
        self.assertFalse(sent)
        attachment.unlink()

        pending = Email._pending_outbox()
        self.assertEqual(len(pending), 1)
        message, meta = Email._read_outbox(pending[0])
        body, files = parts(message)
        self.assertIn('body text', body)
        self.assertEqual(files, [b'alpha,1\n'])
        self.assertEqual(meta['confirmation_message'], 'Autorun')
        self.assertEqual(meta['attempts'], 0)

    def test_next_send_flushes_queued_mail_before_the_new_one(self) -> None:
        with patch.object(Email, '_deliver', side_effect=ConnectionResetError('reset')):
            self.assertFalse(Email.send('Old', 'old body', 'a@b.com', confirmation_message='OldMail'))

        subjects: list[str] = []

        def deliver(message: Message, recipient: str | None = None, timeout: int = 20) -> None:
            del recipient, timeout
            subjects.append(str(message['Subject']))

        with patch.object(Email, '_deliver', side_effect=deliver):
            sent = Email.send('New', 'new body', 'a@b.com', confirmation_message='NewMail')

        self.assertTrue(sent)
        self.assertEqual(len(subjects), 2)
        self.assertIn('Old', subjects[0])
        self.assertIn('New', subjects[1])
        self.assertEqual(Email._pending_outbox(), [])

    def test_network_failure_stops_after_the_oldest_probe(self) -> None:
        calls = {'n': 0}

        def boom(message: Message, recipient: str | None = None, timeout: int = 20) -> None:
            del message, recipient, timeout
            calls['n'] += 1
            raise TimeoutError('down')

        with patch.object(Email, '_deliver', side_effect=boom):
            Email.send('A', 'a', 'a@b.com')
            Email.send('B', 'b', 'a@b.com')
            calls['n'] = 0
            sent = Email.send('C', 'c', 'a@b.com')

        self.assertFalse(sent)
        self.assertEqual(calls['n'], 1)
        subjects = [str(Email._read_outbox(folder)[0]['Subject']) for folder in Email._pending_outbox()]
        self.assertEqual(len(subjects), 3)
        self.assertTrue(all(any(label in subject for subject in subjects) for label in ('A', 'B', 'C')))

    def test_machine_that_cannot_email_does_not_write_the_outbox(self) -> None:
        deliver = Mock(side_effect=AssertionError('smtp'))
        with patch.object(MACHINE, 'emailable', False), patch.object(Email, '_deliver', deliver):
            sent = Email.send('X', 'y', 'a@b.com', attachments=Path(self._tmp.name) / 'missing.csv')
        self.assertFalse(sent)
        deliver.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_repeated_recipient_refusal_moves_mail_to_dead(self) -> None:
        refusal = smtplib.SMTPRecipientsRefused({'a@b.com': (550, b'rejected')})
        with patch.object(Email, '_deliver', side_effect=refusal):
            self.assertFalse(Email.send('Bad', 'body', 'a@b.com'))
        pending = Email._pending_outbox()
        self.assertEqual(len(pending), 1)
        self.assertEqual(Email._read_outbox(pending[0])[1]['attempts'], 1)

        with patch.object(Email, '_deliver', side_effect=refusal):
            for _ in range(3):
                self.assertIsNone(Email._flush_outbox())
                self.assertEqual(len(Email._pending_outbox()), 1)
            self.assertIsNone(Email._flush_outbox())

        self.assertEqual(Email._pending_outbox(), [])
        dead = list((self.root / 'dead').iterdir())
        self.assertEqual(len(dead), 1)
        meta = json.loads((dead[0] / 'meta.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['attempts'], 5)
        self.assertTrue((dead[0] / 'message.eml').is_file())

    def test_caller_owned_retry_delivers_notice_once(self) -> None:
        from src.api.task_monitor import cli_recovery

        state = Path(self._tmp.name) / 'recovery.json'
        cli_recovery.save(state, {'alert_pending': {'error': 'unavailable'}})
        with patch.object(Email, '_deliver', side_effect=TimeoutError('down')):
            for _ in range(2):
                self.assertFalse(cli_recovery.deliver(Email.send, state))
                self.assertEqual(Email._pending_outbox(), [])
                self.assertIn('alert_pending', cli_recovery.load(state))
        with patch.object(Email, '_deliver') as deliver:
            self.assertTrue(cli_recovery.deliver(Email.send, state))
            self.assertTrue(cli_recovery.deliver(Email.send, state))
        deliver.assert_called_once()
        self.assertNotIn('alert_pending', cli_recovery.load(state))

    def test_caller_owned_retry_does_not_enqueue_when_old_mail_blocks(self) -> None:
        with patch.object(Email, '_deliver', side_effect=TimeoutError('down')) as deliver:
            self.assertFalse(Email.send('Old', 'body', 'a@b.com'))
            self.assertFalse(Email.send('Owned by caller', 'body', 'a@b.com', queue_on_failure=False))
        self.assertEqual(deliver.call_count, 2)
        self.assertEqual(len(Email._pending_outbox()), 1)
        with patch.object(Email, '_deliver') as deliver:
            self.assertTrue(Email.send('Owned by caller', 'body', 'a@b.com', queue_on_failure=False))
        self.assertEqual(deliver.call_count, 2)
        self.assertEqual(Email._pending_outbox(), [])

    def test_identical_independent_notifications_are_not_suppressed(self) -> None:
        with patch.object(Email, '_deliver', side_effect=TimeoutError('down')):
            for _ in range(2):
                self.assertFalse(Email.send('Repeated report', 'body', 'a@b.com'))
        with patch.object(Email, '_deliver') as deliver:
            self.assertTrue(Email.send('New', 'body', 'a@b.com'))
        self.assertEqual(deliver.call_count, 3)

    def test_smtp_submission_succeeds_despite_cleanup_errors(self) -> None:
        # Use the real SMTP.quit implementation; no sockets are opened.
        for port in (25, 465, 587):
            for quit_error in (None, TimeoutError('QUIT timeout'), smtplib.SMTPResponseException(500, b'QUIT failed')):
                with self.subTest(port=port, error=type(quit_error).__name__):
                    server = smtplib.SMTP.__new__(smtplib.SMTP)
                    server.starttls = Mock()
                    server.login = Mock()
                    server.send_message = Mock(return_value={})
                    server.docmd = Mock(return_value=(221, b'bye'), side_effect=quit_error)
                    server.close = Mock(side_effect=OSError('close failed') if quit_error else None)
                    with patch.object(Email, 'smtp_port', port), \
                         patch('smtplib.SMTP', return_value=server) as plain, \
                         patch('smtplib.SMTP_SSL', return_value=server) as secure:
                        self.assertTrue(Email.send('Accepted', 'body', 'a@b.com'))
                    self.assertEqual(secure.call_count, int(port == 465))
                    self.assertEqual(plain.call_count, int(port != 465))
                    self.assertEqual(server.starttls.call_count, int(port != 465))
                    server.login.assert_called_once_with(Email.sender, Email.password)
                    server.send_message.assert_called_once()
                    self.assertEqual(server.send_message.call_args.kwargs['to_addrs'], 'a@b.com')
                    server.close.assert_called()
                    self.assertEqual(Email._pending_outbox(), [])

    def test_cleanup_error_preserves_original_submission_failure(self) -> None:
        refusal = smtplib.SMTPRecipientsRefused({'a@b.com': (550, b'rejected')})
        server = Mock()
        server.send_message.side_effect = refusal
        server.quit.side_effect = TimeoutError('QUIT timed out')
        with patch.object(Email, 'smtp_port', 465), patch('smtplib.SMTP_SSL', return_value=server):
            self.assertFalse(Email.send('Rejected', 'body', 'a@b.com'))
        server.close.assert_called_once()
        _, meta = Email._read_outbox(Email._pending_outbox()[0])
        self.assertEqual(meta['attempts'], 1)
        self.assertEqual(meta['last_error'], str(refusal))

    def test_corrupt_entries_are_isolated_without_blocking_other_mail(self) -> None:
        malformed = ('{invalid', '[]', '{"attempts": "bad"}', '{"attempts": -1}')
        for index, raw in enumerate(malformed):
            folder = self.root / f'bad-meta-{index}'
            folder.mkdir(parents=True)
            (folder / 'message.eml').write_bytes(Email.message('Bad', 'body', 'a@b.com').as_bytes())
            (folder / 'meta.json').write_text(raw)
        missing = self.root / 'missing-message'
        missing.mkdir()
        (missing / 'meta.json').write_text('{}')
        broken = self.root / 'broken-message'
        broken.mkdir()
        (broken / 'meta.json').write_text('{}')
        (broken / 'message.eml').write_bytes(b'not a MIME message')
        old = Email._enqueue(Email.message('Valid old', 'body', 'a@b.com'), '', 'offline')
        with patch.object(Email, '_deliver') as deliver:
            self.assertTrue(Email.send('New', 'body', 'a@b.com'))
        self.assertEqual(deliver.call_count, 2)
        self.assertFalse(old.exists())
        self.assertEqual(Email._pending_outbox(), [])
        self.assertEqual(len(list((self.root / 'dead').iterdir())), 6)
        for index, raw in enumerate(malformed):
            self.assertEqual((self.root / 'dead' / f'bad-meta-{index}' / 'meta.json').read_text(), raw)

    def test_failed_quarantine_does_not_block_new_mail(self) -> None:
        folder = self.root / 'bad'
        folder.mkdir(parents=True)
        (folder / 'meta.json').write_text('invalid')
        with patch.object(Email, '_move_dead', side_effect=PermissionError('denied')), \
             patch.object(Email, '_deliver') as deliver:
            self.assertTrue(Email.send('New', 'body', 'a@b.com'))
        deliver.assert_called_once()
        self.assertTrue(folder.exists())

    def test_busy_lock_still_preserves_failed_mail(self) -> None:
        @contextmanager
        def busy():
            yield False

        with patch.object(Email, '_outbox_lock', busy), \
             patch.object(Email, '_deliver', side_effect=TimeoutError('down')):
            self.assertFalse(Email.send('New', 'body', 'a@b.com'))
        self.assertEqual(len(Email._pending_outbox()), 1)

    def test_missing_attachment_warning_and_filename_round_trip(self) -> None:
        attachment = Path(self._tmp.name) / '结果; report.csv'
        attachment.write_bytes(b'alpha,1\n')
        missing = Path(self._tmp.name) / 'missing.csv'
        with patch.object(Email, '_deliver', side_effect=TimeoutError('down')):
            Email.send('Daily', 'body', 'a@b.com', attachments=[attachment, missing], title_prefix=None)
        message, _ = Email._read_outbox(Email._pending_outbox()[0])
        body, files = parts(message)
        self.assertIn(f'Attachment not found: {missing}', body)
        self.assertEqual(files, [b'alpha,1\n'])
        self.assertEqual(message.get_payload()[0].get_filename(), attachment.name)
        self.assertEqual(str(message['Subject']), 'Daily')
