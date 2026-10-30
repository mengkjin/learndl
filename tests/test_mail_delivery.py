"""Size boundaries, real archives, legacy migration and durable failure recovery."""
import io
import base64
import sys
import json
import os
import shutil
import smtplib
import subprocess
import tempfile
import unittest
import zipfile
from email import message_from_bytes, policy
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.proj.util.web import emailer, mail_package as package
from src.proj.util.web.mail_queue import inspect_queue, read_json

Email = emailer.Email


def message(subject='报告', body='正文', attachments=()):
    msg = EmailMessage()
    msg['From'] = 'sender@example.invalid'
    msg['To'] = 'receiver@example.invalid'
    msg['Subject'] = subject
    msg.set_content(body)
    for name, data in attachments:
        msg.add_attachment(data, maintype='application', subtype='octet-stream', filename=name)
    return msg


def attachments(directory, manifest):
    out = {}
    for item in manifest['messages']:
        raw = (directory/item['file']).read_bytes()
        assert len(raw) == item['mime_bytes']
        assert len(raw) <= manifest['max_mime_bytes']
        assert package.digest(raw) == item['sha256']
        msg = message_from_bytes(raw, policy=policy.default)
        assert package.wire(msg) == raw
        for part in msg.walk():
            if part.get_filename():
                out[part.get_filename()] = part.get_payload(decode=True)
    return out


class PackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_small_mime_and_boundary_are_preserved(self):
        msg = message(attachments=[('中文.pdf', os.urandom(20000))])
        msg['Message-ID'] = '<boundary@example.invalid>'
        limit = len(package.wire(msg))
        result = package.prepare(msg, self.root/'exact', 'test', limit)
        self.assertFalse(result['transformed'])
        self.assertEqual((self.root/'exact'/result['messages'][0]['file']).read_bytes(), package.wire(msg))
        smaller = package.prepare(msg, self.root/'smaller', 'test', limit-1)
        self.assertTrue(smaller['transformed'])
        attachments(self.root/'smaller', smaller)

    def test_final_headers_and_raw_groups_are_bounded(self):
        files = [(f'{i}-中文.pdf', os.urandom(18000)) for i in range(4)]
        result = package.prepare(message(attachments=files), self.root, 'group', 40000)
        extracted = attachments(self.root, result)
        self.assertGreater(len(result['messages']), 1)
        for name, data in files:
            self.assertEqual(extracted[name], data)
        for i, item in enumerate(result['messages'], 1):
            msg = message_from_bytes((self.root/item['file']).read_bytes(), policy=policy.default)
            self.assertIn(f'[{i}/{len(result["messages"])}]', str(msg['Subject']))

    def test_compressible_html_is_a_standard_zip(self):
        data = b'<html><p>report data</p></html>' * 100000
        result = package.prepare(message(attachments=[('report.html', data)]), self.root, 'html', 50000)
        files = attachments(self.root, result)
        zipped = next(value for name, value in files.items() if name.endswith('.zip'))
        with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
            self.assertEqual(archive.read('report.html'), data)

    def test_long_body_preserved_without_manual_reassembly(self):
        body = '完整正文测试\n' * 100000
        result = package.prepare(message(body=body), self.root, 'body', 80000)
        files = attachments(self.root, result)
        zipped = next(value for name, value in files.items() if name.endswith('.zip'))
        with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
            text = archive.read(archive.namelist()[0]).decode()
            self.assertEqual(text.replace('\r\n', '\n'), body)
        for item in result['messages']:
            self.assertLess(item['mime_bytes'], 80000)

    def test_related_html_preserves_cid_and_resources(self):
        msg = EmailMessage()
        msg['To'] = 'receiver@example.invalid'
        msg['Subject'] = 'related'
        msg.set_content('<img src="cid:image">' + 'content '*10000, subtype='html')
        msg.add_related(b'picture', maintype='image', subtype='png', cid='<image>')
        result = package.prepare(msg, self.root, 'related', 50000)
        files = attachments(self.root, result)
        zipped = next(value for name, value in files.items() if name.endswith('.zip'))
        with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
            restored = message_from_bytes(archive.read(archive.namelist()[0]), policy=policy.default)
        self.assertEqual(restored.get_content_type(), 'multipart/related')
        self.assertEqual(list(restored.iter_parts())[1]['Content-ID'], '<image>')
        self.assertEqual(list(restored.iter_parts())[1].get_payload(decode=True), b'picture')

    def test_missing_tool_is_explicit_and_source_unchanged(self):
        msg = message(attachments=[('large.pdf', os.urandom(100000))])
        original = package.wire(msg)
        with patch.object(package.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, '7-Zip missing'):
                package.prepare(msg, self.root, 'missing', 30000)
        self.assertEqual(package.wire(msg), original)

    @unittest.skipUnless(shutil.which('7zz') or shutil.which('7z'), '7-Zip integration requires 7zz/7z')
    def test_real_7zip_volumes_roundtrip_and_hashes(self):
        data = os.urandom(160000)
        result = package.prepare(message(attachments=[('中文报告.pdf', data)]), self.root/'mail', 'real', 40000)
        files = attachments(self.root/'mail', result)
        destination = self.root/'download'
        destination.mkdir()
        for name, raw in files.items():
            (destination/name).write_bytes(raw)
        first = next(destination.glob('*.7z.001'))
        subprocess.run([shutil.which('7zz') or shutil.which('7z'), 'x', str(first), f'-o{destination}/out'],
                       check=True, capture_output=True)
        actual = (destination/'out/中文报告.pdf').read_bytes()
        self.assertEqual(actual, data)
        self.assertEqual(package.digest(actual), result['attachments'][0]['sha256'])
        sizes = [part['size'] for part in result['attachments'][0]['volumes']]
        self.assertLessEqual(max(sizes)-min(sizes), 1)
        for part in result['attachments'][0]['volumes']:
            self.assertEqual(package.digest(files[part['name']]), part['sha256'])

    def test_duplicate_names_and_path_escape_are_safe(self):
        files = [('same.txt', b'a'*30000), ('same.txt', b'b'*30000), ('../../unsafe.txt', b'c'*30000)]
        result = package.prepare(message(attachments=files), self.root, 'names', 50000)
        found = attachments(self.root, result)
        self.assertEqual(found['same.txt'], files[0][1])
        self.assertEqual(found['2-same.txt'], files[1][1])
        self.assertEqual(found['unsafe.txt'], files[2][1])

    def test_balanced_volume_sizes_and_capacity(self):
        self.assertEqual(package.balanced_sizes(45_000_000, 20_000_000), [15_000_000]*3)
        self.assertEqual(package.balanced_sizes(40_000_000, 20_000_000), [20_000_000]*2)
        lengths = package.balanced_sizes(40_000_001, 20_000_000)
        self.assertEqual(sum(lengths), 40_000_001)
        self.assertLessEqual(max(lengths)-min(lengths), 1)
        self.assertEqual(package.attachment_capacity(29_900_000), 20_000_000)
        budget = 39000
        cap = package.attachment_capacity(budget)
        for size, fits in ((cap, True), (cap+1, False)):
            encoded = 4*((size+2)//3)
            self.assertEqual(encoded + 2*((encoded+75)//76) <= budget, fits)

    def test_attachment_limit_applies_even_when_whole_mail_fits(self):
        msg = message(attachments=[('report.html', b'x'*20_000_001)])
        self.assertLess(len(package.wire(msg)), package.MAX_BYTES)
        result = package.prepare(msg, self.root, 'attachment-limit')
        self.assertTrue(result['transformed'])
        files = attachments(self.root, result)
        self.assertTrue(any(name.endswith('.zip') for name in files))
        self.assertTrue(all(len(raw) <= 20_000_000 for raw in files.values()))

    def test_actual_30mb_limit(self):
        msg = message(attachments=[('near-limit.bin', b'x'*21_930_000)])
        raw = package.wire(msg)
        self.assertGreater(len(raw), package.MAX_BYTES)
        result = package.prepare(msg, self.root, '30mb')
        self.assertTrue(result['transformed'])
        attachments(self.root, result)


class QueueTest(unittest.TestCase):
    def test_long_filename_real_task_mail_and_legacy_receipt(self):
        from src.proj.util.web.task_mail import TaskResultMail, retry_delivery
        paths = []
        for suffix in ('.json', '.csv', '.log'):
            path = self.root/f'rank_scatter_20261030_111402_554012{suffix}'
            path.write_text('benchmark results\n')
            paths.append(path)
        def transport(msg, recipient):
            key = package.digest(b'long-filenames')
            folder = self.root/'.external'/key
            if not folder.exists():
                folder.mkdir(parents=True)
                (folder/'message.eml').write_bytes(msg.as_bytes())
                package.atomic_json(folder/'meta.json', {'attempts': 0})
                stored, _ = Email._read_outbox(folder)
                # The previous CRLF-body-only fix still fails on these real names.
                self.assertNotEqual(package.fingerprint(stored, normalize_body_newlines=True),
                                    package.fingerprint(msg, normalize_body_newlines=True))
            return Email.send_with_smtplib(msg, recipient, delivery_id='long-filenames')
        directory = self.root/'long-task.mail'
        sender = TaskResultMail(directory, email_api=Email, transport=transport)
        self.assertTrue(sender('NN rank benchmark - success', 'Results: 202\nValidation: exact\n',
                               'test@example.invalid', attachments=paths))
        self.assertTrue(retry_delivery(directory, transport=transport))
        self.assertEqual(self.deliver.call_count, 1)

    def test_folded_headers_preserve_attachment_name_protection(self):
        long_name = 'rank_scatter_20261030_111402_554012.json'
        first = message_from_bytes(package.wire(message(attachments=[(long_name, b'{}')])))
        other = message_from_bytes(package.wire(message(attachments=[(long_name+'.changed', b'{}')])))
        self.assertTrue(Email.send_with_smtplib(first, delivery_id='name-protection'))
        self.assertFalse(Email.send_with_smtplib(other, delivery_id='name-protection'))
        self.assertIn('headers', str(Email.logger.error.call_args))
        self.assertEqual(self.deliver.call_count, 1)

    def test_task_result_mail_through_external_receipt(self):
        from src.proj.util.web.task_mail import TaskResultMail, retry_delivery
        attachment = self.root/'benchmark.log'
        attachment.write_text('completed\n')
        def transport(msg, recipient):
            return Email.send_with_smtplib(msg, recipient, delivery_id='benchmark-result')
        sender = TaskResultMail(self.root/'task-mail', email_api=Email, transport=transport)
        self.assertTrue(sender('Benchmark', 'status: success\nrows: 202\n',
                               'test@example.invalid', attachments=[attachment]))
        self.assertTrue(retry_delivery(self.root/'task-mail', transport=transport))
        self.assertEqual(self.deliver.call_count, 1)

    def test_external_smtp_body_newlines_and_existing_failed_receipt(self):
        incoming = message_from_bytes(package.wire(message(body='first line\nsecond line\n',
            attachments=[('result.csv', b'a,b\r\n1,2\r\n')])))
        # Reproduce a receipt created by the old implementation: local as_bytes
        # converts the unencoded inline body from CRLF to LF before reading it.
        key = package.digest(b'old-failed-delivery')
        folder = self.root/'.external'/key
        folder.mkdir(parents=True)
        (folder/'message.eml').write_bytes(incoming.as_bytes())
        package.atomic_json(folder/'meta.json', {'attempts': 0, 'subject': 'report'})
        stored = message_from_bytes((folder/'message.eml').read_bytes())
        self.assertNotEqual(package.fingerprint(incoming), package.fingerprint(stored))
        self.assertTrue(Email.send_with_smtplib(incoming, delivery_id='old-failed-delivery'))
        self.assertEqual(self.deliver.call_count, 1)
        self.assertTrue(Email.send_with_smtplib(incoming, delivery_id='old-failed-delivery'))
        self.assertEqual(self.deliver.call_count, 1)  # Already delivered; never resend.

    def test_external_still_rejects_changed_body_or_attachment(self):
        original = message(body='first\nsecond\n', attachments=[('result.csv', b'a\r\nb')])
        self.assertTrue(Email.send_with_smtplib(original, delivery_id='immutable'))
        for changed in (message(body='first\nCHANGED\n', attachments=[('result.csv', b'a\r\nb')]),
                        message(body='first\nsecond\n', attachments=[('result.csv', b'a\nb')])):
            self.assertFalse(Email.send_with_smtplib(changed, delivery_id='immutable'))
        self.assertEqual(self.deliver.call_count, 1)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.enterContext(patch.object(emailer, 'PATH', SimpleNamespace(email_outbox=self.root)))
        self.enterContext(patch.object(emailer, 'MACHINE', SimpleNamespace(emailable=True)))
        self.enterContext(patch.object(Email, 'logger', Mock()))
        self.enterContext(patch.object(Email, 'max_message_bytes', 50000))
        self.deliver = self.enterContext(patch.object(Email, '_deliver'))
        self.now = [1_800_000_000.0]
        self.enterContext(patch('src.proj.util.web.mail_queue.time.time', side_effect=lambda: self.now[0]))

    def enqueue(self, msg=None):
        return Email._enqueue(msg or message(), '', '')

    def advance(self, seconds=3601):
        self.now[0] += seconds

    def test_legacy_large_message_migrates_after_small_mail_and_keeps_original(self):
        large = self.enqueue(message('large', attachments=[('report.html', b'<p>data</p>'*50000)]))
        original = (large/'message.eml').read_bytes()
        small = self.enqueue(message('small'))
        def deliver(msg, **kwargs):
            self.assertLessEqual(len(package.wire(msg)), 50000)
        self.deliver.side_effect = deliver
        stats = Email.retry_pending()
        self.assertEqual(str(self.deliver.call_args_list[0].args[0]['Subject']), 'small')
        self.assertEqual(stats['remaining'], 0)
        self.assertEqual((self.root/'.archive'/large.name/'message.eml').read_bytes(), original)
        self.assertTrue((self.root/'.archive'/small.name).exists())
        calls = self.deliver.call_count
        Email.retry_pending()
        self.assertEqual(calls, self.deliver.call_count)

    def test_missing_tool_isolates_large_mail_but_delivers_small(self):
        large = self.enqueue(message('large', attachments=[('a.pdf', os.urandom(100000))]))
        self.enqueue(message('small'))
        with patch.object(package.shutil, 'which', return_value=None):
            Email.retry_pending()
        self.assertTrue((self.root/'dead'/large.name/'message.eml').is_file())
        self.assertEqual(self.deliver.call_count, 1)
        self.assertEqual(Email._pending_outbox(), [])
        self.assertIn('7-Zip', next(iter(read_json(self.root/'.notices.json').values()))['error'])

    def test_submission_disconnect_skips_bad_and_counts_only_that_mail(self):
        bad = self.enqueue(message('bad'))
        self.enqueue(message('good'))
        def deliver(msg, **kwargs):
            if str(msg['Subject']) == 'bad':
                error = smtplib.SMTPServerDisconnected('disconnected during DATA')
                error.mail_stage = 'submit'
                raise error
        self.deliver.side_effect = deliver
        stats = Email.retry_pending()
        self.assertEqual((stats['attempted'], stats['sent']), (2, 1))
        item = read_json(bad/'payload/manifest.json')['messages'][0]
        self.assertEqual(item['attempts'], 1)
        calls = self.deliver.call_count
        Email.retry_pending()
        self.assertEqual(calls, self.deliver.call_count)
        self.advance(61)
        Email.retry_pending()
        self.assertEqual(read_json(bad/'payload/manifest.json')['messages'][0]['attempts'], 2)

    def test_connection_cooldown_and_recovery_do_not_consume_item_attempts(self):
        folders = [self.enqueue(message(str(i))) for i in range(3)]
        self.deliver.side_effect = ConnectionError('offline')
        stats = Email.retry_pending()
        self.assertEqual(stats['attempted'], 2)
        self.assertFalse((folders[2]/'payload').exists())
        self.assertEqual(read_json(folders[0]/'payload/manifest.json')['messages'][0]['attempts'], 0)
        self.assertTrue(Email._queue().cooling())
        self.deliver.reset_mock(side_effect=True)
        Email.retry_pending()
        self.deliver.assert_not_called()
        self.advance(61)
        stats = Email.retry_pending()
        self.assertEqual(stats['sent'], 3)

    def test_partial_batch_restart_only_sends_remaining_and_external_receipt_is_stable(self):
        msg = message(attachments=[(f'{i}.pdf', os.urandom(22000)) for i in range(3)])
        calls = []
        def deliver(msg, **kwargs):
            calls.append(str(msg['Message-ID']))
            if len(calls) == 2:
                raise smtplib.SMTPDataError(451, b'try later')
        self.deliver.side_effect = deliver
        self.assertFalse(Email.send_with_smtplib(msg, delivery_id='report-1'))
        first = list(calls)
        self.advance()
        self.assertTrue(Email.send_with_smtplib(message(attachments=[('unused', b'')]), delivery_id='other'))
        self.assertTrue(Email.send_with_smtplib(msg, delivery_id='report-1'))
        self.assertEqual(calls.count(first[0]), 1)
        self.assertEqual(calls.count(first[1]), 2)
        self.assertEqual(Email._pending_outbox(), [])
        self.assertFalse(Email.send_with_smtplib(message(body='changed'), delivery_id='report-1'))

    def test_notices_delayed_coalesced_non_recursive_and_recovery(self):
        self.enqueue(message('bad'))
        self.deliver.side_effect = smtplib.SMTPDataError(451, b'try later')
        Email.retry_pending()
        self.advance(301)
        self.deliver.side_effect = None
        Email.retry_pending()
        subjects = [str(c.args[0]['Subject']) for c in self.deliver.call_args_list]
        self.assertEqual(sum('Mail delivery status' in s for s in subjects), 1)
        self.assertEqual(Email._pending_outbox(), [])
        self.advance(1801)
        Email.retry_pending()
        subjects = [str(c.args[0]['Subject']) for c in self.deliver.call_args_list]
        self.assertEqual(sum('Mail delivery status' in s for s in subjects), 2)
        self.assertIn('recovered', ''.join(p.get_payload(decode=True).decode() for p in self.deliver.call_args.args[0].walk() if not p.is_multipart()))
        calls = self.deliver.call_count
        self.advance(3601)
        Email.retry_pending()
        self.assertEqual(calls, self.deliver.call_count)

    def test_failed_warning_waits_and_does_not_create_recursive_notice(self):
        queue = Email._queue()
        queue.notice('bad', subject='bad', size=100, status='isolated', error='bad attachment')
        error = smtplib.SMTPDataError(451, b'notice failed')
        error.mail_stage = 'submit'
        self.deliver.side_effect = error
        Email.retry_pending()
        self.assertEqual(len(read_json(self.root/'.notices.json')), 1)
        calls = self.deliver.call_count
        self.deliver.side_effect = None
        self.enqueue(message('good'))
        Email.retry_pending()
        self.assertEqual(self.deliver.call_count, calls+1)  # good mail can continue
        self.advance(301)
        Email.retry_pending()
        self.assertTrue(read_json(self.root/'.notice_delivery.json')['complete'])

    def test_preparation_publish_recovers_without_duplicate(self):
        folder = self.enqueue(message('migrate', attachments=[('html', b'x'*100000)]))
        queue = Email._queue()
        initial = queue.prepare_folder(folder)
        with patch('src.proj.util.web.mail_queue.prepare', side_effect=AssertionError('must reuse')):
            self.assertEqual(queue.prepare_folder(folder), initial)
            Email.retry_pending()
        self.assertEqual(read_json(self.root/'.archive'/folder.name/'payload/manifest.json')['fingerprint'], initial['fingerprint'])

    def test_corrupt_prepared_manifest_isolated_without_blocking_others(self):
        folder = self.enqueue(message('corrupt'))
        Email._queue().prepare_folder(folder)
        package.atomic_json(folder/'payload/manifest.json', {'messages': []})
        self.enqueue(message('good'))
        Email.retry_pending()
        self.assertTrue((self.root/'dead'/folder.name/'message.eml').exists())
        self.assertEqual(self.deliver.call_count, 1)
        self.assertEqual(str(self.deliver.call_args.args[0]['Subject']), 'good')

    def test_disk_failure_preserves_source_and_no_partial_publish(self):
        folder = self.enqueue(message())
        original = (folder/'message.eml').read_bytes()
        with patch.object(package, 'atomic_bytes', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                Email._queue().prepare_folder(folder)
        self.assertEqual((folder/'message.eml').read_bytes(), original)
        self.assertFalse((folder/'payload').exists())

    def test_auth_failure_cools_without_consuming_single_mail_attempts(self):
        folder = self.enqueue(message())
        error = smtplib.SMTPAuthenticationError(535, b'authentication failed')
        error.mail_stage = 'auth'
        self.deliver.side_effect = error
        result = Email.retry_pending()
        self.assertEqual(result['status'], 'cooldown')
        self.assertEqual(result['attempted'], 1)
        self.assertEqual(read_json(folder/'payload/manifest.json')['messages'][0]['attempts'], 0)

    def test_five_submission_failures_isolate_and_manual_retry_keeps_success(self):
        folder = self.enqueue(message('bad'))
        def deliver(msg, **kwargs):
            if str(msg['Subject']) == 'bad':
                raise smtplib.SMTPDataError(451, b'temporary rejection')
        self.deliver.side_effect = deliver
        for _ in range(5):
            Email.retry_pending()
            self.advance()
        isolated = self.root/'dead'/folder.name
        self.assertTrue(isolated.exists())
        self.assertEqual(read_json(isolated/'payload/manifest.json')['messages'][0]['attempts'], 5)
        from src.proj.util.web.mail_queue import main
        with patch.object(sys, 'argv', ['mail_queue', 'retry', folder.name]):
            main()
        self.assertTrue(folder.exists())
        self.deliver.side_effect = None
        Email.retry_pending()
        self.assertFalse(folder.exists())
        self.assertTrue((self.root/'.archive'/folder.name).exists())

    def test_notice_pages_resume_without_resending_accepted_pages(self):
        queue = Email._queue()
        for i in range(60):
            queue.notice(str(i), subject='bad', size=100, status='isolated', error='x'*1900)
        counter = [0]
        def deliver(msg, **kwargs):
            counter[0] += 1
            if counter[0] == 2:
                error = smtplib.SMTPServerDisconnected('warning interrupted')
                error.mail_stage = 'submit'
                raise error
        self.deliver.side_effect = deliver
        Email.retry_pending()
        initial = read_json(self.root/'.notice_delivery.json')
        self.assertTrue(initial['pages'][0]['sent'])
        self.assertFalse(initial['complete'])
        first_id = self.deliver.call_args_list[0].args[0]['Message-ID']
        self.advance(301)
        Email.retry_pending()
        self.assertTrue(read_json(self.root/'.notice_delivery.json')['complete'])
        self.assertEqual(sum(c.args[0]['Message-ID'] == first_id for c in self.deliver.call_args_list), 1)
        self.assertEqual(len(read_json(self.root/'.notices.json')), 60)

    def test_legacy_148m_migration_in_fresh_process(self):
        # Stream a real >140 MiB legacy MIME file instead of allocating it in the
        # parent. The child exercises the same reader as the 1 GiB watchdog.
        folder = self.root/'legacy-large'
        folder.mkdir()
        block = base64.encodebytes(b'<p>training log data</p>\n'*256)
        with (folder/'message.eml').open('wb') as stream:
            stream.write(b'To: test@example.invalid\nSubject: legacy large HTML\nMIME-Version: 1.0\n'
                         b'Content-Type: multipart/mixed; boundary="boundary"\n\n'
                         b'--boundary\nContent-Type: text/html\nContent-Disposition: attachment; filename="large.html"\n'
                         b'Content-Transfer-Encoding: base64\n\n')
            for _ in range(18000):
                stream.write(block)
            stream.write(b'\n--boundary--\n')
        package.atomic_json(folder/'meta.json', {'attempts': 0})
        code = """
import json,resource,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from src.proj.util.web import emailer
from src.api.task_monitor import watchdog
with patch.object(emailer,'PATH',SimpleNamespace(email_outbox=Path(sys.argv[1]))),patch.object(emailer,'MACHINE',SimpleNamespace(emailable=True)),patch.object(emailer.Email,'logger',Mock()),patch.object(emailer.Email,'_deliver'):
    result=emailer.Email.retry_pending(max_seconds=60)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak=peak if sys.platform=='darwin' else peak*1024
    print(json.dumps(dict(result=result,peak_bytes=peak)))
"""
        result = subprocess.run([sys.executable, '-c', code, str(self.root)], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        print(f'Large-mail watchdog regression peak: {output["peak_bytes"]/1024**2:.1f} MiB')
        self.assertEqual(output['result']['remaining'], 0)
        self.assertEqual(output['result']['sent'], 1)
        self.assertLess(output['peak_bytes'], 1024**3, output)
        backup = self.root/'.archive'/folder.name
        self.assertTrue((backup/'message.eml').exists())
        self.assertGreater((backup/'message.eml').stat().st_size, 140*1024**2)
        self.assertLess(read_json(backup/'payload/manifest.json')['messages'][0]['mime_bytes'], 30_000_000)

    def test_read_only_inspection_does_not_write_or_send(self):
        before = list(self.root.iterdir())
        result = inspect_queue(Email)
        self.assertEqual(result['entries'], [])
        self.assertEqual(list(self.root.iterdir()), before)
        self.deliver.assert_not_called()


class SMTPTest(unittest.TestCase):
    def test_wire_size_and_submission_error_stage(self):
        msg = message(attachments=[('file.txt', b'abc')])
        server = Mock()
        error = smtplib.SMTPServerDisconnected('lost')
        server.sendmail.side_effect = error
        server.quit.side_effect = smtplib.SMTPServerDisconnected('cleanup')
        with patch.object(Email, 'smtp_port', 465), patch.object(emailer.smtplib, 'SMTP_SSL', return_value=server):
            with self.assertRaises(smtplib.SMTPServerDisconnected) as caught:
                Email._deliver(msg)
        self.assertEqual(caught.exception.mail_stage, 'submit')
        self.assertEqual(server.sendmail.call_args.args[2], package.wire(msg))

    def test_overlimit_never_connects_and_authentication_is_distinct(self):
        msg = message(body='x'*10000)
        with patch.object(Email, 'max_message_bytes', 1000), patch.object(emailer.smtplib, 'SMTP_SSL') as connect:
            with self.assertRaises(ValueError):
                Email._deliver(msg)
            connect.assert_not_called()
        server = Mock()
        server.login.side_effect = smtplib.SMTPAuthenticationError(535, b'bad credentials')
        with patch.object(Email, 'smtp_port', 465), patch.object(emailer.smtplib, 'SMTP_SSL', return_value=server):
            with self.assertRaises(smtplib.SMTPAuthenticationError) as caught:
                Email._deliver(message())
        self.assertEqual(caught.exception.mail_stage, 'auth')


if __name__ == '__main__':
    unittest.main()
