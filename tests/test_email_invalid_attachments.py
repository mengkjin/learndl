"""Invalid attachments must not prevent task notifications from reaching SMTP."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

from src.proj.env.variable.files import EmailAttachment
from src.proj.util.web import emailer

__all__ = ['InvalidAttachmentTests']

Email = emailer.Email


class InvalidAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.object(emailer, 'PATH', SimpleNamespace(email_outbox=self.root / 'outbox')))
        self.enterContext(patch.object(emailer, 'MACHINE', SimpleNamespace(emailable=True)))
        self.logger = self.enterContext(patch.object(Email, 'logger', Mock()))
        self.deliver = self.enterContext(patch.object(Email, '_deliver'))

    def sent_message(self):
        self.deliver.assert_called_once()
        return self.deliver.call_args.args[0]

    def body(self, message):
        return '\n'.join(part.get_payload(decode=True).decode('utf-8')
                         for part in message.walk() if part.get_content_type() == 'text/plain')

    def test_directory_and_missing_path_warn_but_valid_attachment_is_sent(self):
        valid = self.root / 'report.csv'
        valid.write_text('date,value\n1,2\n')
        missing = self.root / 'missing.csv'
        self.assertTrue(Email.send('Completed', 'Original summary', 'nobody@example.invalid',
                                  attachments=[self.root, missing, EmailAttachment(valid, 'result.csv')]))
        message = self.sent_message()
        parts = [p for p in message.walk() if p.get_content_disposition() == 'attachment']
        self.assertEqual([p.get_filename() for p in parts], ['result.csv'])
        self.assertEqual(parts[0].get_payload(decode=True), valid.read_bytes())
        self.assertIn('Original summary', self.body(message))
        self.assertIn('Warning: Attachment skipped:', self.body(message))
        self.assertEqual(self.logger.warning.call_count, 2)
        self.logger.error.assert_not_called()

    def test_all_invalid_still_sends_body_and_project_attachment_warnings(self):
        queue = Mock()
        queue.pop_all.return_value = [EmailAttachment(self.root)]
        with patch.object(emailer, 'Proj', SimpleNamespace(email_attachments=queue)):
            self.assertTrue(Email.send('Complete', 'Data ready', 'nobody@example.invalid',
                                      project_attachments=True))
        message = self.sent_message()
        self.assertFalse(any(p.get_content_disposition() == 'attachment' for p in message.walk()))
        self.assertIn('Data ready', self.body(message))
        self.logger.warning.assert_called_once()
        queue.pop_all.assert_called_once()

    def test_unreadable_or_disappearing_file_does_not_abort_later_attachments(self):
        bad, good = self.root / 'bad.txt', self.root / 'good.txt'
        bad.write_text('bad')
        good.write_text('good')
        original_open = Path.open
        for error in (PermissionError('denied'), FileNotFoundError('deleted during read')):
            with self.subTest(error=type(error).__name__):
                self.deliver.reset_mock()

                def open_file(path, *args, **kwargs):
                    if path == bad:
                        raise error
                    return original_open(path, *args, **kwargs)

                with patch.object(Path, 'open', open_file):
                    self.assertTrue(Email.send('Ready', 'summary', 'nobody@example.invalid',
                                              attachments=[bad, good]))
                message = self.sent_message()
                self.assertEqual([p.get_filename() for p in message.walk() if p.get_filename()], ['good.txt'])
                self.assertIn(type(error).__name__, self.body(message))

    def test_invalid_spec_and_filename_are_skipped(self):
        bad = self.root / 'bad\nname.txt'
        bad.write_text('not for MIME')
        items: Any = [None, bad]
        self.assertTrue(Email.send('Ready', 'summary', 'nobody@example.invalid', attachments=items))
        message = self.sent_message()
        self.assertEqual(self.logger.warning.call_count, 2)
        self.assertFalse(any(p.get_filename() for p in message.walk()))


if __name__ == '__main__':
    unittest.main()
