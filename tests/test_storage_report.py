"""Disk reporting checks with temporary data and no real email delivery."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.proj.util.filesys.storage_report import scan_storage, render_html, write_report
from src.api.calls.computer import ProjectStorageReport


class StorageReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

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
        if sys.platform.startswith('linux'):
            du = subprocess.check_output(['du', '-s', '-B1', str(self.root)], text=True)
            self.assertEqual(scan.node.size, int(du.split()[0]))

    def test_errors_propagate_without_aborting(self):
        self.file('blocked/a')
        self.file('good/a')
        original = os.scandir

        def scandir(path):
            if Path(path).name == 'blocked':
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
             patch('src.proj.util.cli.prompts.prompt_text', side_effect=['', '']):
            result = ProjectStorageReport.go(email=False)
        self.assertEqual(result['scan'].root, self.root.resolve())
        self.assertIn(('Project Storage Report', ProjectStorageReport),
                      [(label, call) for label, call, _ in DirectCallHub._source_code_entries()])


if __name__ == '__main__':
    unittest.main()
