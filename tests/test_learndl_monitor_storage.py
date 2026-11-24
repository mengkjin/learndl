"""Exercise storage UI APIs, incremental progress and email with temporary data."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.api.learndl_monitor import storage
from src.proj.util.filesys.storage_browser import StorageReportServer
from src.proj.util.filesys.storage_report import generate_report

__all__ = []


class StorageServiceTest(unittest.TestCase):
    def test_progress_snapshot_email_and_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'sample.txt').write_text('test')
            _, report = generate_report(root, root / 'snapshot.html')
            from src.automation.artifact_exchange.storage_insights import embed_metadata, metadata
            remote = root / 'share/persistent/storage_insights/server/latest.html'
            remote.parent.mkdir(parents=True)
            remote.write_text(embed_metadata(report.read_text(), dict(metadata(report.read_text()), source_host='server')))
            servers = []
            ready = threading.Event()
            entered = threading.Event()
            release = threading.Event()
            calls = []

            def report_run(instance):
                calls.append(instance.kwargs['email'])
                instance.kwargs['progress']('Scanned 100 entries: sample')
                entered.set()
                self.assertTrue(release.wait(10))
                return {'report': report, 'email_sent': instance.kwargs['email']}

            # Reserve a port, then keep it bound through the server factory.
            reserved = ThreadingHTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler)
            port = reserved.server_port
            reserved.server_close()

            def fixed_server(address, handler):
                server = ThreadingHTTPServer(address, handler)
                servers.append(server)
                ready.set()
                return server

            def snapshot_server(path):
                server = StorageReportServer(path)
                servers.append(server)
                return server

            with patch('src.api.calls.computer.ProjectStorageReport.run', report_run), \
                    patch('src.automation.artifact_exchange.core.Config.load', return_value=SimpleNamespace(shared_root=root / 'share', enabled=False, role='receiver')), \
                    patch('src.automation.artifact_exchange.storage_insights.publish_daily_safe') as publisher, \
                    patch('src.proj.PATH', SimpleNamespace(production=root)), \
                    patch('src.proj.util.web.emailer.Email.send', return_value=True) as email, \
                    patch('src.proj.util.filesys.storage_browser.StorageReportServer', snapshot_server), \
                    patch.object(storage, 'STATE', root / 'state'), \
                    patch.dict(storage.SERVICES, storage=dict(storage.SERVICES['storage'], port=port)), \
                    patch.object(storage, 'ThreadingHTTPServer', fixed_server):
                worker = threading.Thread(target=storage.main, daemon=True)
                worker.start()
                try:
                    self.assertTrue(ready.wait(10))
                    record = root / 'state/storage.json'
                    for _ in range(100):
                        if record.exists():
                            break
                        time.sleep(.01)
                    address = json.loads(record.read_text())['url']
                    origin = f'http://127.0.0.1:{port}'

                    def status():
                        with urlopen(address + '/status', timeout=5) as response:
                            return json.load(response)

                    def post(action, body=b'{}', request_origin=origin):
                        return urlopen(Request(address + '/' + action, data=body,
                                               headers={'Origin': request_origin}), timeout=5)

                    self.assertFalse(status()['busy'])
                    self.assertEqual(calls, [])
                    self.assertEqual(status()['sources'][0]['source_host'], 'server')
                    with urlopen(address + '/remote/server', timeout=5) as response:
                        self.assertIn('sandbox allow-scripts', response.headers['Content-Security-Policy'])
                        remote_page = response.read()
                        self.assertIn(b'Project Storage Report', remote_page)
                        self.assertIn(b'color-scheme: dark', remote_page)
                        self.assertIn(b'learndl-storage-language', remote_page)
                    self.assertEqual(calls, [])
                    with post('scan', b'{"email":false}') as response:
                        self.assertEqual(response.status, 202)
                    self.assertTrue(entered.wait(10))
                    self.assertTrue(status()['busy'])
                    self.assertEqual(status()['progress'], ['Scanned 100 entries: sample'])
                    with self.assertRaises(HTTPError) as conflict:
                        post('scan')
                    self.assertEqual(conflict.exception.code, 409)
                    with self.assertRaises(HTTPError) as forbidden:
                        post('scan', request_origin='http://example.org')
                    self.assertEqual(forbidden.exception.code, 403)
                    release.set()
                    for _ in range(100):
                        if not status()['busy']:
                            break
                        time.sleep(.02)
                    self.assertEqual(calls, [False])
                    with urlopen(address + '?lang=en', timeout=5) as response:
                        page = response.read()
                        self.assertIn(b'emailAfter', page)
                        self.assertIn(b'role="tablist"', page)
                    with urlopen(address + '/snapshot/', timeout=5) as response:
                        snapshot_page = response.read()
                        self.assertIn(b'/snapshot/', snapshot_page)
                        self.assertIn(b'color-scheme: dark', snapshot_page)
                        self.assertIn(b'learndl-storage-language', snapshot_page)
                    self.assertNotIn('color-scheme:dark', report.read_text())
                    with post('scan') as response:
                        self.assertEqual(response.status, 202)
                    for _ in range(100):
                        if not status()['busy']:
                            break
                        time.sleep(.02)
                    self.assertEqual(calls, [False, True])
                    self.assertEqual(publisher.call_count, 2)
                    publisher.assert_called_with(root, 0, force=True, report=report)
                    with post('email') as response:
                        self.assertEqual(response.status, 202)
                    for _ in range(100):
                        if not status()['busy']:
                            break
                        time.sleep(.02)
                    email.assert_called_once()
                    self.assertEqual(email.call_args.kwargs['attachments'], [report])
                finally:
                    release.set()
                    for server in servers:
                        server.shutdown()
                        server.server_close()
                    worker.join(5)


if __name__ == '__main__':
    unittest.main()
