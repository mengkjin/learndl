"""Disk reporting checks with temporary data and no real email delivery."""
import os
import subprocess
import sys
import tempfile
import sqlite3
import threading
import urllib.request
import urllib.error
import json
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

from src.proj.util.filesys.storage_report import scan_storage, render_html, write_report, generate_report
from src.proj.util.filesys.storage_browser import StorageReportServer
from src.api.calls.computer import ProjectStorageReport


class StorageReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('stdin', 'stdout'):
            tty = patch(f'sys.{name}.isatty', return_value=False)
            tty.start()
            self.addCleanup(tty.stop)

    def file(self, relative, content=b'x' * 8192):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def nodes(self, node):
        return {node.path: node} | {key: value for child in node.children for key, value in self.nodes(child).items()}

    def test_depths_totals_and_hidden_files(self):
        for path in ('data/DataBase/DB_trade_ts/min/2025/a', 'models/nn/gru/archive/0/a',
                     'data/Interim/DataBlock/fit/2025/a', 'data/Export/stock_factor/beta/2025/a',
                     'results/trade/backtest/run/a', '.logs/model/operation/year/a',
                     '.local_resources/server/cache/deep/a', 'src/package/deep/a', '.git/objects/a'):
            self.file(path)
        self.file('root-file')
        (self.root / 'empty').mkdir()
        scan = scan_storage(self.root)
        nodes = self.nodes(scan.node)
        for expected in ('data/DataBase/DB_trade_ts/min', 'models/nn/gru',
                         'data/Interim/DataBlock/fit', 'data/Export/stock_factor/beta',
                         'results/trade/backtest', '.logs/model/operation',
                         '.local_resources/server/cache', 'src/package', '.git/objects', 'empty'):
            self.assertIn(expected, nodes)
        self.assertNotIn('models/nn/gru/archive', nodes)
        self.assertNotIn('data/DataBase/DB_trade_ts/min/2025', nodes)
        expected = self.root.stat().st_blocks * 512 + sum(p.lstat().st_blocks * 512 for p in self.root.rglob('*'))
        self.assertEqual(scan.node.size, expected)
        for node in nodes.values():
            self.assertEqual(node.size, node.own + sum(child.size for child in node.children))

    def test_sparse_hardlinks_and_symlinks(self):
        path = self.file('a/data')
        (self.root / 'b').mkdir()
        os.link(path, self.root / 'b' / 'duplicate')
        (self.root / 'link').symlink_to(self.root, target_is_directory=True)
        (self.root / 'broken').symlink_to(self.root / 'absent')
        with (self.root / 'sparse').open('wb') as stream:
            stream.truncate(64 * 1024 * 1024)
        scan = scan_storage(self.root)
        nodes = self.nodes(scan.node)
        self.assertEqual(nodes['b'].size, (self.root / 'b').stat().st_blocks * 512)
        self.assertEqual(nodes['link'].size, (self.root / 'link').lstat().st_blocks * 512)
        self.assertFalse(nodes['link'].children)
        self.assertLess(scan.node.size, 64 * 1024 * 1024)
        self.assertTrue(scan.node.complete)
        self.assertEqual(scan.node.file_count, 3)
        self.assertEqual(nodes['b'].file_count, 1)
        if sys.platform.startswith('linux'):
            du = subprocess.check_output(['du', '-s', '-B1', str(self.root)], text=True)
            self.assertEqual(scan.node.size, int(du.split()[0]))

    def test_recursive_file_counts_and_date_ranges_in_snapshot(self):
        folder = 'data/DataBase/trade_ts/day'
        for name in ('2024/day.20240229.feather', '2025/day.20250102.feather',
                     '2025/day.20250230.feather', 'notes.txt'):
            self.file(f'{folder}/{name}')
        (self.root / folder / '20991231').mkdir()
        (self.root / 'empty').mkdir()
        scan, report = generate_report(self.root, self.root / 'report.html')
        node = self.nodes(scan.node)[folder]
        self.assertEqual(node.file_count, 4)
        self.assertEqual((node.date_start, node.date_end), ('2024-02-29', '2025-01-02'))
        self.assertEqual(scan.node.file_count, 4)
        self.assertEqual(self.nodes(scan.node)['empty'].file_count, 0)
        self.assertIsNone(self.nodes(scan.node)['empty'].date_start)
        self.assertIn('4 files · Dates: 2024-02-29 → 2025-01-02', report.read_text())
        with StorageReportServer(report) as server:
            rendered = server.children(node.node_id)['html']
            self.assertIn('2 files · Dates: 2025-01-02 → 2025-01-02', rendered)
            self.assertIn('0 files', rendered)
            self.assertNotIn('Dates: 2099', rendered)

    def test_errors_propagate_without_aborting(self):
        self.file('blocked/a')
        self.file('good/a')
        original = os.scandir

        def scandir(path):
            if not isinstance(path, int) and Path(path).name == 'blocked':
                raise PermissionError('test denied')
            return original(path)

        with patch('os.scandir', side_effect=scandir):
            scan = scan_storage(self.root)
        self.assertFalse(scan.node.complete)
        self.assertEqual(scan.error_count, 1)
        self.assertIn('good', self.nodes(scan.node))
        self.assertFalse(self.nodes(scan.node)['blocked'].complete)

    def test_html_escaping_and_exclusive_output(self):
        (self.root / '<script>alert("x")<').mkdir()
        scan = scan_storage(self.root)
        rendered = render_html(scan)
        self.assertIn('&lt;script&gt;', rendered)
        self.assertNotIn('<script>alert', rendered)
        self.assertIn('id="search"', rendered)
        self.assertIn('id="sort"', rendered)
        self.assertNotIn('src="https:', rendered)
        output = write_report(scan, self.root / 'report.html')
        self.assertEqual(output.read_text(), rendered)
        with self.assertRaises(FileExistsError):
            write_report(scan, output)

    def test_disappearing_file_marks_ancestors_incomplete(self):
        path = self.file('data/DataBase/test/dataset/gone').resolve()
        original = Path.lstat

        def lstat(candidate):
            if candidate == path:
                raise FileNotFoundError('disappeared during scan')
            return original(candidate)

        with patch.object(Path, 'lstat', lstat):
            scan = scan_storage(self.root)
        self.assertEqual(scan.error_count, 1)
        self.assertFalse(scan.node.complete)
        self.assertFalse(self.nodes(scan.node)['data/DataBase/test/dataset'].complete)

    def test_unsupported_allocated_metric(self):
        from types import SimpleNamespace
        original = Path.stat

        def no_blocks(path, **kwargs):
            info = original(path, **kwargs)
            return SimpleNamespace(st_mode=info.st_mode)

        with patch.object(Path, 'stat', no_blocks), self.assertRaisesRegex(RuntimeError, 'st_blocks'):
            scan_storage(self.root)

    def test_default_email_attachment_and_failure(self):
        from src.proj.util.web.emailer import Email
        for outcome in (True, False, RuntimeError('simulated SMTP failure')):
            with self.subTest(outcome=outcome), patch.object(Email, 'send') as send:
                if isinstance(outcome, Exception):
                    send.side_effect = outcome
                else:
                    send.return_value = outcome
                result = ProjectStorageReport.go(root=self.root)
                self.assertTrue(result['report'].is_file())
                self.assertTrue(result['report'].with_suffix('.sqlite3').is_file())
                self.assertEqual(result['email_sent'], outcome is True)
                self.assertEqual(send.call_args.kwargs['attachments'], [result['report']])
                self.assertIsNone(send.call_args.kwargs['recipient'])
                self.assertFalse(send.call_args.kwargs['project_attachments'])

    def test_email_disabled_and_explicit_recipient(self):
        from src.proj.util.web.emailer import Email
        with patch.object(Email, 'send', return_value=True) as send:
            result = ProjectStorageReport.go(root=self.root, output='offline.html', email=False)
            send.assert_not_called()
            self.assertIsNone(result['email_sent'])
            ProjectStorageReport.go(root=self.root, recipient='test@example.invalid')
            self.assertEqual(send.call_args.kwargs['recipient'], 'test@example.invalid')

    def test_interactive_defaults_and_menu(self):
        from src.api.calls import computer
        from src.api.calls.launcher import DirectCallHub
        from types import SimpleNamespace
        with patch.object(computer, 'PATH', SimpleNamespace(production=self.root)), \
             patch('sys.stdin.isatty', return_value=True), patch('sys.stdout.isatty', return_value=True), \
             patch('src.proj.util.cli.prompts.prompt_text') as path_prompt, \
             patch('src.proj.util.cli.prompts.prompt_confirm', return_value=False) as confirm:
            result = ProjectStorageReport.go(email=False)
        confirm.assert_called_once()
        path_prompt.assert_not_called()
        self.assertEqual(result['scan'].root, self.root.resolve())
        self.assertIn(('Project Storage Report', ProjectStorageReport),
                      [(label, call) for label, call, _ in DirectCallHub._source_code_entries()])

    def snapshot(self):
        self.file('models/nn/gru/archive/0/2025/a')
        self.file('models/nn/gru/snapshot/a')
        (self.root / 'models/nn/gru/empty').mkdir()
        (self.root / 'models/nn/gru/link').symlink_to(self.root, target_is_directory=True)
        return generate_report(self.root, self.root / 'report.html')

    def test_deep_snapshot_and_no_rescan(self):
        scan, report = self.snapshot()
        default = self.nodes(scan.node)
        self.assertNotIn('models/nn/gru/archive', default)
        self.assertTrue(default['models/nn/gru'].has_children)
        with StorageReportServer(report) as server:
            rows = {row['path']: row for row in server.connection.execute('SELECT * FROM nodes')}
            self.assertIn('models/nn/gru/archive/0/2025', rows)
            for path, node in default.items():
                self.assertEqual(rows[path]['size'], node.size)
            parent = rows['models/nn/gru']
            # Mutating a file after generation must never change the indexed result.
            self.file('models/nn/gru/archive/0/2025/a', b'changed')
            result = server.children(parent['node_id'])['html']
            self.assertIn('models/nn/gru/archive', result)
            self.assertNotIn('models/nn/gru/archive/0', result)
            self.assertIn('class="load-next"', result)
            self.assertEqual(result, server.children(parent['node_id'])['html'])
            for leaf in ('empty', 'link'):
                self.assertFalse(rows[f'models/nn/gru/{leaf}']['has_children'])
            self.assertTrue(rows['models/nn/gru/snapshot']['has_children'])
            files = server.children(rows['models/nn/gru/snapshot']['node_id'])['html']
            self.assertIn('models/nn/gru/snapshot/a', files)
            self.assertIn('class="node file"', files)
            self.assertNotIn('<details', files)
            self.assertNotIn('load-next', files)
            self.assertNotIn('本层占用', report.read_text())
            self.assertIn('const storageBase = null;', report.read_text())
            self.assertNotIn('const storageBase = null;', server.page.decode())

    def test_index_hardlinks_errors_and_safe_rendering(self):
        first = self.file('models/nn/gru/a/first')
        second = self.root / 'models/nn/gru/b'
        second.mkdir()
        os.link(first, second / 'duplicate')
        self.file('models/nn/gru/blocked/a')
        self.file('models/nn/gru/<img onerror="bad()">/a')
        original = os.scandir

        def scandir(path):
            if not isinstance(path, int) and Path(path).name == 'blocked':
                raise PermissionError('simulated')
            return original(path)

        with patch('os.scandir', side_effect=scandir):
            scan, report = generate_report(self.root, self.root / 'report.html')
        with StorageReportServer(report) as server:
            rows = {row['path']: row for row in server.connection.execute('SELECT * FROM nodes')}
            self.assertEqual(rows['models/nn/gru/b']['size'], second.stat().st_blocks * 512)
            self.assertEqual(rows['models/nn/gru/b/duplicate']['size'], 0)
            self.assertFalse(rows['models/nn/gru']['complete'])
            rendered = server.children(rows['models/nn/gru']['node_id'])['html']
            self.assertIn('&lt;img', rendered)
            self.assertNotIn('<img', rendered)
            self.assertIn('Incomplete', rendered)
        self.assertFalse(scan.node.complete)

    def test_reject_missing_corrupt_mismatched_and_foreign_index(self):
        scan = scan_storage(self.root)
        old = write_report(scan, self.root / 'old.html')
        with self.assertRaisesRegex(ValueError, 'Regenerate'):
            StorageReportServer(old)
        _, report = self.snapshot()
        with patch('src.proj.util.filesys.storage_browser.machine_identity', return_value='foreign'):
            with self.assertRaisesRegex(ValueError, 'computer'):
                StorageReportServer(report)
        original = report.read_bytes()
        report.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'match'):
            StorageReportServer(report)
        report.write_bytes(original)
        report.with_suffix('.sqlite3').write_bytes(b'corrupt')
        with self.assertRaises(sqlite3.DatabaseError):
            StorageReportServer(report)

    def test_failed_generation_publishes_no_report(self):
        output = self.root / 'report.html'
        with patch('src.proj.util.filesys.storage_browser.SnapshotIndex.add', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                generate_report(self.root, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.sqlite3').exists())
        self.assertFalse(list(self.root.glob('.storage-stage-*')))
        original_link = os.link

        def link(source, destination):
            if Path(destination) == output:
                raise OSError('publication failed')
            return original_link(source, destination)

        with patch('os.link', side_effect=link), self.assertRaises(OSError):
            generate_report(self.root, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix('.sqlite3').exists())

    def test_http_scope_and_read_only_queries(self):
        _, report = self.snapshot()
        with StorageReportServer(report) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(server.url) as response:
                    self.assertIn('Local browsing', response.read().decode())
                    self.assertIsNone(response.headers.get('Access-Control-Allow-Origin'))
                with urllib.request.urlopen(server.url + 'children?id=1') as response:
                    self.assertEqual(json.load(response)['node_id'], 1)
                for suffix in ('children?id=999999', 'children?id=../etc/passwd',
                               'children?id=1&path=/etc/passwd', 'children?id=1&id=2',
                               '../report.html', 'index.sqlite3', 'children?id=0'):
                    with self.subTest(suffix=suffix), self.assertRaises(urllib.error.HTTPError) as raised:
                        urllib.request.urlopen(server.url + suffix)
                    self.assertEqual(raised.exception.code, 404)
                for headers in ({'Host': 'foreign.invalid'}, {'Origin': 'https://foreign.invalid'},
                                {'Sec-Fetch-Site': 'cross-site'}):
                    request = urllib.request.Request(server.url, headers=headers)
                    with self.assertRaises(urllib.error.HTTPError) as raised:
                        urllib.request.urlopen(request)
                    self.assertEqual(raised.exception.code, 403)
                with self.assertRaises(sqlite3.OperationalError):
                    server.connection.execute('DELETE FROM nodes')
            finally:
                server.shutdown()
                thread.join(timeout=3)

    def test_idle_browser_connection_does_not_block_loading(self):
        from src.proj.util.filesys.storage_browser import _ReportHandler
        _, report = self.snapshot()
        accepted = threading.Event()
        setup = _ReportHandler.setup

        def ready(handler):
            setup(handler)
            accepted.set()

        with StorageReportServer(report) as server, patch.object(_ReportHandler, 'setup', ready):
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            idle = socket.create_connection(server.server_address, timeout=2)
            try:
                self.assertTrue(accepted.wait(2))
                with urllib.request.urlopen(server.url + 'children?id=1', timeout=2) as response:
                    self.assertEqual(json.load(response)['node_id'], 1)
            finally:
                idle.close()
                server.shutdown()
                thread.join(timeout=3)

    def test_hidden_entries_remain_in_snapshot_and_totals(self):
        for path in ('.hidden/a', '~backup/a', 'normal/.file', 'normal/~file', 'normal/visible'):
            self.file(path)
        scan, report = generate_report(self.root, self.root / 'report.html')
        nodes = self.nodes(scan.node)
        self.assertIn('.hidden', nodes)
        self.assertIn('normal/~file', nodes)
        self.assertEqual(scan.node.size, scan.node.own + sum(n.size for n in scan.node.children))
        page = report.read_text()
        self.assertIn('id="hide-special" type="checkbox" checked', page)
        self.assertIn('applySearch();', page)
        self.assertNotIn('AbortSignal.timeout', page)

    def test_browse_cli_stops_and_closes_service(self):
        from src.api.calls.launcher import DirectCallHub
        _, report = self.snapshot()
        with patch.object(StorageReportServer, 'serve_forever', side_effect=KeyboardInterrupt), \
             patch('webbrowser.open_new_tab', return_value=True) as opened:
            ProjectStorageReport._open_local_report(report)
        self.assertTrue(opened.call_args.args[0].startswith('http://127.0.0.1:'))
        self.assertNotIn('Browse Storage Report', [label for label, _, _ in DirectCallHub._source_code_entries()])

    def test_interactive_prompt_before_scan_and_open_after_email(self):
        from src.proj.util.web.emailer import Email
        from src.proj.util.filesys.storage_report import generate_report as generate
        events = []
        def record_scan(*args, **kwargs):
            events.append('scan')
            return generate(*args, **kwargs)
        with patch('sys.stdin.isatty', return_value=True), patch('sys.stdout.isatty', return_value=True), \
             patch('src.proj.util.filesys.storage_report.generate_report', side_effect=record_scan), \
             patch.object(Email, 'send', side_effect=lambda *a, **kw: events.append('email') or False), \
             patch('src.proj.util.cli.prompts.prompt_confirm', side_effect=lambda *a, **kw: events.append('prompt') or True), \
             patch.object(ProjectStorageReport, '_open_local_report', side_effect=lambda p: events.append('open')):
            result = ProjectStorageReport.go(root=self.root)
        self.assertEqual(events, ['prompt', 'scan', 'email', 'open'])
        self.assertTrue(result['report'].is_file())
        with patch('src.proj.util.cli.prompts.prompt_confirm') as confirm, \
             patch('src.proj.util.cli.prompts.prompt_text') as prompt, \
             patch.object(ProjectStorageReport, '_open_local_report') as opened:
            ProjectStorageReport.go(root=self.root, email=False)
        confirm.assert_not_called()
        prompt.assert_not_called()
        opened.assert_not_called()

    def test_static_file_rows(self):
        self.file('readme.txt')
        self.file('src/file<name>.py')
        scan = scan_storage(self.root)
        nodes = self.nodes(scan.node)
        self.assertIn('readme.txt', nodes)
        self.assertIn('src/file<name>.py', nodes)
        page = render_html(scan)
        self.assertIn('file&lt;name&gt;.py', page)
        self.assertIn('class="node file"', page)
        self.assertIn('lang="en"', page)
        self.assertNotRegex(page, r'[\u3400-\u4dbf\u4e00-\u9fff]')

    def test_cancel_initial_prompt_does_not_scan(self):
        with patch('sys.stdin.isatty', return_value=True), patch('sys.stdout.isatty', return_value=True), \
             patch('src.proj.util.cli.prompts.prompt_confirm', return_value=None), \
             patch('src.proj.util.filesys.storage_report.generate_report') as generate:
            self.assertIsNone(ProjectStorageReport.go(root=self.root, email=False))
        generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
