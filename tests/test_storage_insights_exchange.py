"""Static storage report publication uses isolated shares and no email."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.automation.artifact_exchange.storage_insights import publish_daily, metadata, ReportCatalog
from src.proj.util.filesys.storage_report import scan_storage, render_html


class StorageInsightsExchangeTest(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'project'
        self.root.mkdir()
        (self.root / 'data.txt').write_text('test')
        self.config = SimpleNamespace(enabled=True, role='publisher', shared_root=self.base / 'share',
                                      state=self.base / 'private', source_host='server')
        self.now = datetime.now(timezone.utc)
        self.target = self.base / 'share/persistent/storage_insights/server/latest.html'
        self.addCleanup(patch.stopall)
        patch('src.automation.artifact_exchange.storage_insights.sandboxed', return_value=False).start()

    def publish(self, now=None):
        return publish_daily(self.root, 0, config=self.config, now=now or self.now)

    def test_daily_dedup_replace_and_static_only(self):
        self.assertEqual(self.publish(), 'published')
        before = self.target.read_bytes()
        stamp = self.target.stat().st_mtime_ns
        with patch('src.proj.util.filesys.storage_report.scan_storage', side_effect=AssertionError('Duplicate scan')):
            self.assertEqual(self.publish(), 'already published')
        self.assertEqual(self.target.stat().st_mtime_ns, stamp)
        self.assertEqual(self.publish(self.now + timedelta(days=1)), 'published')
        self.assertNotEqual(self.target.read_bytes(), before)
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])
        self.assertEqual(list(self.base.rglob('*.sqlite3')), [])

    def test_failure_preserves_last_report_then_retries(self):
        self.publish()
        before = self.target.read_bytes()
        tomorrow = self.now + timedelta(days=1)
        with patch('src.proj.util.filesys.storage_report.scan_storage', side_effect=OSError('scan failed')):
            with self.assertRaises(OSError):
                self.publish(tomorrow)
        self.assertEqual(self.target.read_bytes(), before)
        original = Path.replace
        def fail_replace(path, destination):
            if Path(destination) == self.target:
                raise OSError('publish interrupted')
            return original(path, destination)
        with patch.object(Path, 'replace', fail_replace):
            with self.assertRaises(OSError):
                self.publish(tomorrow)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])
        self.assertEqual(self.publish(tomorrow), 'published')

    def test_reuses_sufficiently_recent_manual_report(self):
        from src.proj.env.path import storage_reports
        scan = scan_storage(self.root)
        folder = storage_reports(self.root)
        folder.mkdir(parents=True)
        (folder / 'manual.html').write_text(render_html(scan))
        with patch('src.proj.util.filesys.storage_report.scan_storage', side_effect=AssertionError('Repeated scan')):
            self.assertEqual(self.publish(), 'published')
        self.assertEqual(metadata(self.target.read_text())['report_id'], scan.report_id)

    def test_multiple_hosts_cached_and_partial_reports(self):
        scan = scan_storage(self.root)
        scan.node.complete = False
        scan.error_count = 1
        with patch('src.proj.util.filesys.storage_report.scan_storage', return_value=scan):
            self.publish()
        self.config.source_host = 'second'
        self.publish()
        catalog = ReportCatalog(self.config.shared_root)
        entries = catalog.refresh()
        self.assertEqual({report['source_host'] for report in entries}, {'server', 'second'})
        self.assertFalse(next(report for report in entries if report['source_host'] == 'server')['complete'])
        with patch.object(Path, 'read_text', side_effect=AssertionError('Unchanged report read again')):
            self.assertEqual(catalog.refresh(), entries)
        self.assertEqual(metadata(catalog.entries['server']['content'])['root'], str(self.root.resolve()))

    def test_receiver_and_autofix_skip(self):
        self.config.role = 'receiver'
        self.assertEqual(self.publish(), 'disabled')
        self.config.role = 'publisher'
        with patch('src.automation.artifact_exchange.storage_insights.sandboxed', return_value=True):
            self.assertEqual(self.publish(), 'disabled')
        self.assertFalse(self.target.exists())

    def test_concurrent_requests_scan_once(self):
        import threading
        entered, release = threading.Event(), threading.Event()
        actual = scan_storage
        results = []
        def slow_scan(root):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test timed out')
            return actual(root)
        with patch('src.proj.util.filesys.storage_report.scan_storage', side_effect=slow_scan) as scanner:
            worker = threading.Thread(target=lambda: results.append(self.publish()))
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.publish(), 'publication in progress')
            finally:
                release.set()
                worker.join(5)
            self.assertEqual(results, ['published'])
            self.assertEqual(scanner.call_count, 1)

    def test_manual_report_before_update_completion_requires_fresh_scan(self):
        from src.proj.env.path import storage_reports
        scan = scan_storage(self.root)
        folder = storage_reports(self.root)
        folder.mkdir(parents=True)
        (folder / 'older.html').write_text(render_html(scan))
        completed = datetime.fromisoformat(scan.finished).timestamp() + 1
        publish_daily(self.root, completed, config=self.config, now=self.now)
        self.assertNotEqual(metadata(self.target.read_text())['report_id'], scan.report_id)

    def test_daily_hook_skips_incomplete_update_and_report_failure_is_contained(self):
        import ast
        import sys
        from unittest.mock import Mock
        import src.proj
        source = Path('scripts/1_autorun/0_daily_update.py').read_text()
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
        function.decorator_list = []
        namespace = {'SharedSync': SimpleNamespace(sync=Mock())}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<daily-update-test>', 'exec'), namespace)
        daily = Mock()
        updated = Mock(return_value=False)
        with patch.dict(sys.modules, {'src.api.pkgs.update': SimpleNamespace(UpdateAPI=SimpleNamespace(daily=daily)),
                                      'src.api.pkgs.data': SimpleNamespace(DataAPI=SimpleNamespace(is_updated=updated))}), \
             patch.object(src.proj, 'MACHINE', SimpleNamespace(updatable=True)), \
             patch('src.automation.artifact_exchange.storage_insights.publish_daily_safe') as publisher:
            namespace['main']()
            publisher.assert_not_called()
            updated.return_value = True
            namespace['main']()
            publisher.assert_called_once()
        from src.automation.artifact_exchange.storage_insights import publish_daily_safe
        with patch('src.automation.artifact_exchange.storage_insights.publish_daily', side_effect=OSError('failed')), \
             patch.object(src.proj.Logger, 'error') as error:
            self.assertIsNone(publish_daily_safe(self.root, 0))
            error.assert_called_once()

    def test_forced_manual_report_replaces_today_without_scanning_again(self):
        self.publish()
        previous = metadata(self.target.read_text())['report_id']
        manual = self.base / 'manual.html'
        scan = scan_storage(self.root)
        manual.write_text(render_html(scan))
        with patch('src.proj.util.filesys.storage_report.scan_storage', side_effect=AssertionError('Repeated scan')):
            result = publish_daily(self.root, 0, config=self.config, now=self.now, force=True, report=manual)
        self.assertEqual(result, 'published')
        self.assertEqual(metadata(self.target.read_text())['report_id'], scan.report_id)
        self.assertNotEqual(previous, scan.report_id)
        self.assertEqual(ReportCatalog(self.config.shared_root).refresh()[0]['report_id'], scan.report_id)
        self.assertFalse((self.config.shared_root / 'storage_insights').exists())
