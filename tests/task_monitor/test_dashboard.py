"""Task Center HTTP and browser regressions with isolated task and log fixtures."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from src.api.task_monitor.actions import KillResult, KillTarget
from src.api.task_monitor.core import TaskPage, TaskSnapshot, WatchdogHealth
from src.api.task_monitor.launch import create_server
from src.api.task_monitor.remote import Publisher
from src.api.task_monitor.web_api import TaskCenter


class DashboardTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / 'run.log'
        self.log.write_text('<span style="color: red">Running output</span>\n')
        self.report = self.root / 'report.html'
        self.report.write_text('<h1>Report fixture</h1><script>window.reportRan = true</script>')
        self.running = self.snapshot('run', 'running', self.log)
        self.finished = self.snapshot('old', 'complete', self.report)
        repo = Mock()
        repo.runtime_dir = self.root / 'runtime'
        repo.get_task.side_effect = {'run': self.running, 'old': self.finished}.get
        repo.list_tasks.side_effect = lambda **params: TaskPage(
            (self.running,) if params['statuses'] == {'running'} else (self.finished,) if params['statuses'] else (),
            int(bool(params['statuses'])), None, params['statuses'] == {'running'})
        repo.status_counts.return_value = {'complete': 1, 'error': 0, 'killed': 0}
        repo.watchdog_health.return_value = WatchdogHealth(None, False, {'disabled': True})
        self.repository = repo
        self.center = TaskCenter(repo, self.root / 'shared')
        self.publisher = Publisher(self.root / 'shared', self.root / 'private', 'test server')
        self.remote_log = self.root / 'remote.log'
        self.remote_log.write_bytes(b'earlier remote log\n' + b'partial line\n' * 5500 +
                               b'<span style="color: red">Remote output</span>\n')
        self.remote_task = dict(task_id='run', script='remote.py', status='running', start_time=time.time(),
                                log=str(self.remote_log))
        self.publisher.tick([self.remote_task])
        self.host = self.publisher.shared.name
        server = create_server(0, center=self.center)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.base = f'http://127.0.0.1:{server.server_port}'
        with urlopen(self.base) as response:
            token = re.search('name="task-token" content="([^"]+)"', response.read().decode())
            assert token is not None
            self.token = token[1]

    @staticmethod
    def snapshot(task_id, status, path):
        now = time.time()
        return TaskSnapshot(task_id=task_id, script=task_id + '.py', cmd='python ' + task_id + '.py',
                            status=status, create_time=now, start_time=now, end_time=now if status == 'complete' else None,
                            source='test', pid=123, exit_code=None, exit_message=None, exit_error=None, exit_files=(path,))

    def request(self, path, *, query=None, body=None, token=True, origin=None):
        headers = {'Authorization': 'Bearer ' + self.token} if token else {}
        if origin is not None or body is not None:
            headers['Origin'] = origin or self.base
        url = self.base + path + ('?' + urlencode(query) if query else '')
        data = json.dumps(body).encode() if body is not None else None
        with urlopen(Request(url, data=data, headers=headers)) as response:
            return response.read()

    def get_json(self, path, **kwargs):
        return json.loads(self.request(path, **kwargs))

    def test_all_deep_links_and_health(self):
        self.assertEqual(self.get_json('/health', token=False)['service'], 'tasks')
        for path in ['/', '/remote', '/finished']:
            page = self.request(path, token=False)
            self.assertIn(b'<script type="module" src="/dashboard.js">', page)
            self.assertIn(b'sandbox=""', page)
        with self.assertRaises(HTTPError) as missing:
            self.request('/api/running', token=False)
        self.assertEqual(missing.exception.code, 403)

    def test_local_running_output_and_finished_filters(self):
        self.assertEqual(self.get_json('/api/running')['tasks'][0]['task_id'], 'run')
        query = {'task': 'run', 'style': 'colored'}
        self.assertIn('<span style="color: red">', self.get_json('/api/output', query=query)['content'])
        query['style'] = 'text'
        self.assertEqual(self.get_json('/api/output', query=query)['content'], 'Running output')
        self.assertEqual(self.get_json('/api/finished')['tasks'][0]['task_id'], 'old')
        self.assertEqual(self.get_json('/api/finished', query={'statuses': ''})['tasks'], [])
        with self.assertRaises(HTTPError) as invalid:
            self.get_json('/api/finished', query={'ended_from': 20, 'ended_before': 10})
        self.assertEqual(invalid.exception.code, 400)

    def test_report_preview_and_download_are_task_scoped(self):
        query = {'task': 'old', 'source': 0}
        self.assertIn('Report fixture', self.get_json('/api/report', query=query)['html'])
        self.assertEqual(self.request('/api/download', query=query), self.report.read_bytes())
        for invalid in ({'task': 'run', 'source': 0}, {'task': 'old', 'source': -1}):
            with self.assertRaises(HTTPError) as missing:
                self.request('/api/download', query=invalid)
            self.assertEqual(missing.exception.code, 404)

    def test_remote_latest_history_styles_and_reload(self):
        self.assertEqual(self.get_json('/api/remote')[0]['id'], self.host)
        query = {'host': self.host, 'task': 'run', 'style': 'colored'}
        detail = self.get_json('/api/remote/task', query=query)
        self.assertTrue(detail['complete'])
        latest = self.get_json('/api/remote/output', query=query)
        self.assertIn('<span style="color: red">Remote output', latest['content'])
        query['before'] = latest['start']
        query['style'] = 'text'
        self.assertTrue(self.get_json('/api/remote/output', query=query)['content'].startswith('earlier remote log'))
        self.assertGreater(self.center.remote_cache.total, 0)
        self.get_json('/api/remote/reload', body={'host': self.host, 'task': 'run'})
        self.assertEqual(self.center.remote_cache.total, 0)

    def test_patch_snapshot_can_arrive_before_the_task_index(self):
        from src.api.task_monitor.patch_progress import FILENAME
        from src.shared.file_ops import atomic_json

        path = self.publisher.shared.parent / 'upgrading' / FILENAME
        progress = dict(host='Upgrading server', status='running', sampled_at=time.time())
        atomic_json(path, progress)
        hosts = {row['id']: row for row in self.get_json('/api/remote')}
        self.assertEqual(hosts['upgrading']['version_patch'], progress)
        self.assertEqual(hosts['upgrading']['tasks'], [])
        self.assertIsNone(hosts['upgrading']['sampled_at'])
        self.assertIsNone(hosts[self.host]['version_patch'])
        self.assertIsNone(self.get_json('/api/running')['version_patch'])

    def test_remote_v1_host_with_same_task_id_loads_its_own_output(self):
        legacy = self.root / 'shared' / 'runtime' / 'legacy'
        from src.api.task_monitor.remote import task_key
        task_dir = legacy / 'tasks' / task_key('run')
        task_dir.mkdir(parents=True)
        (task_dir / '0-00000000.bin').write_bytes(b'legacy output')
        (legacy / 'index.json').write_text(json.dumps({
            'version': 1, 'host': 'legacy server', 'sampled_at': time.time(),
            'tasks': [dict(task_id='run', script='remote.py', status='running',
                           generation=0, offset=13, key=task_dir.name)],
        }))
        query = {'host': self.host, 'task': 'run', 'style': 'text'}
        self.assertIn('Remote output', self.get_json('/api/remote/output', query=query)['content'])
        query['host'] = 'legacy'
        self.assertEqual(self.get_json('/api/remote/output', query=query)['content'], 'legacy output')

    def test_remote_exited_process_can_still_be_inspected_before_final_status(self):
        self.publisher.tick([dict(self.remote_task, pid_alive=False),
                             dict(task_id='queued', script='queued.py', status='starting', phase='queued')])
        tasks = self.get_json('/api/remote')[0]['tasks']
        self.assertEqual({task['task_id']: task['status'] for task in tasks}, {'run': 'stale', 'queued': 'starting'})
        query = {'host': self.host, 'task': 'run', 'style': 'text'}
        self.assertEqual(self.get_json('/api/remote/task', query=query)['status'], 'stale')
        self.assertIn('Remote output', self.get_json('/api/remote/output', query=query)['content'])

        self.publisher.tick([dict(self.remote_task, status='killed', end_time=time.time())])
        self.assertEqual(self.get_json('/api/remote/task', query=query)['status'], 'killed')

    def test_termination_is_confirmed_once_on_same_origin(self):
        target = KillTarget('run', 123, 100, 'run.py', 100)
        with patch('src.api.task_monitor.web_api.prepare_kill', return_value=target), \
             patch('src.api.task_monitor.web_api.kill_task', return_value=KillResult('success', 'Terminated')) as stop:
            prepared = self.get_json('/api/kill/prepare', body={'task': 'run'})
            with self.assertRaises(HTTPError) as forbidden:
                self.request('/api/kill/confirm', body={'token': prepared['token']}, origin='http://example.com')
            self.assertEqual(forbidden.exception.code, 403)
            stop.assert_not_called()
            self.assertEqual(self.get_json('/api/kill/confirm', body={'token': prepared['token']})['status'], 'success')
            with self.assertRaises(HTTPError):
                self.request('/api/kill/confirm', body={'token': prepared['token']})
            stop.assert_called_once_with(target)

    def test_guest_can_confirm_a_termination(self):
        headers = {'Authorization': 'Bearer ' + self.token, 'Origin': self.base,
                   'X-LearnDL-Role': 'admin', 'X-Forwarded-Host': 'phone.example'}
        with urlopen(Request(self.base, headers=headers)) as page:
            self.assertIn(b'data-monitor-role="guest"', page.read())
        def action(path, data):
            with urlopen(Request(self.base + path, headers=headers, data=json.dumps(data).encode())) as result:
                return json.load(result)
        target = KillTarget('run', 123, 100, 'run.py', 100)
        with patch('src.api.task_monitor.web_api.prepare_kill', return_value=target), \
             patch('src.api.task_monitor.web_api.kill_task', return_value=KillResult('success', 'Terminated')) as stop:
            prepared = action('/api/kill/prepare', {'task': 'run'})
            self.assertEqual(action('/api/kill/confirm', {'token': prepared['token']})['status'], 'success')
            stop.assert_called_once_with(target)

    @unittest.skipUnless(os.environ.get('LEARNDL_TEST_BROWSER'), 'set LEARNDL_TEST_BROWSER to run Chrome verification')
    def test_browser_terminal_preferences_and_finished_report(self):
        from playwright.sync_api import expect, sync_playwright
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=os.environ['LEARNDL_TEST_BROWSER'])
            page = browser.new_page(viewport={'width': 1280, 'height': 900}, timezone_id='America/New_York')
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(self.base + '/?lang=en')
            expect(page.locator('h1')).to_have_text('Task Center')
            expect(page.locator('.card')).to_have_count(1)
            expect(page.locator('.output span')).to_have_text('Running output')
            page.locator('#settings summary').click()
            slider = page.get_by_role('slider', name='Terminal font size')
            slider.fill('18')
            expect(page.locator('.live-output')).to_have_css('font-size', '18px')
            page.locator('#style').select_option('text')
            expect(page.locator('.output span')).to_have_count(0)
            phone = browser.new_page(viewport={'width': 375, 'height': 812}, is_mobile=True, has_touch=True)
            phone.goto(self.base)
            phone.locator('#task-list summary').click()
            expect(phone.locator('#cards')).to_be_hidden()
            expect(phone.locator('.live-output')).to_be_visible()
            phone.locator('#task-list summary').click()
            expect(phone.locator('.card')).to_be_visible()
            phone.locator('#settings summary').click()
            settings_panel = phone.locator('#settings .monitor-panel').bounding_box()
            assert settings_panel is not None
            self.assertGreaterEqual(settings_panel['x'], 0)
            self.assertLessEqual(settings_panel['x'] + settings_panel['width'], 375)
            expect(phone.locator('#terminal-size')).to_have_value('12')
            phone.locator('#terminal-size').fill('10')
            expect(phone.locator('.live-output')).to_have_css('font-size', '10px')
            phone.reload()
            phone.locator('#settings summary').click()
            expect(phone.locator('#terminal-size')).to_have_value('10')
            expect(slider).to_have_value('18')
            phone.close()
            self.publisher.tick([self.remote_task, dict(task_id='queued', script='queued.py',
                                                       status='starting', phase='queued', start_time=1)])
            page.locator('nav a[href="/remote"]').click()
            expect(page.locator('#style')).to_have_value('text')
            expect(page.locator('#queue-title')).to_have_text('Running (1) · Queued (1)')
            from src.api.task_monitor.patch_progress import FILENAME
            from src.shared.file_ops import atomic_json

            progress = dict(host='test server', status='running', applied='0.7.1', target='0.7.2',
                            patch='0.7.2/001', script='001_update.py', remaining=2, attempt=1,
                            started=time.time() - 180, sampled_at=time.time())
            path = self.publisher.shared / FILENAME
            atomic_json(path, progress)
            page.locator('#refresh-page').click()
            expect(page.locator('#patch-phase')).to_contain_text('Applying patch')
            expect(page.locator('#patch-detail')).to_contain_text('Patches remaining 2')
            atomic_json(path, dict(progress, sampled_at=time.time() - 90))
            page.locator('#refresh-page').click()
            expect(page.locator('#patch-phase')).to_contain_text('Last reported running')
            atomic_json(path, dict(host='test server', status='ready', applied='0.7.2', target='0.7.2',
                                   remaining=0, sampled_at=time.time()))
            page.locator('#refresh-page').click()
            expect(page.locator('#patch-phase')).to_have_text('Version ready')
            expect(page.locator('.card[data-task-id="queued"]')).to_contain_text('Queued')
            expect(page.locator('.output')).to_contain_text('Remote output')
            viewport = page.locator('.live-output')
            expect(viewport).to_have_css('font-size', '18px')
            page.locator('#settings summary').click()
            slider.fill('20')
            self.assertTrue(viewport.evaluate('(node) => node.scrollHeight - node.clientHeight - node.scrollTop <= 2'))
            viewport.evaluate('(node) => { node.scrollTop = 100; }')
            expect(page.get_by_role('button', name='Jump to bottom')).to_be_visible()
            first_visible = """node => Array.from(node.firstElementChild.children)
              .findIndex(line => line.offsetTop + line.offsetHeight > node.scrollTop)"""
            anchor = viewport.evaluate(first_visible)
            slider.fill('18')
            self.assertEqual(viewport.evaluate(first_visible), anchor)
            page.locator('#settings summary').click()
            with self.remote_log.open('ab') as stream:
                stream.write(b'New remote output\n')
            self.publisher.tick([self.remote_task])
            page.get_by_role('button', name='Jump to bottom').click()
            expect(page.locator('.output')).to_contain_text('New remote output')
            page.locator('#earlier').click()
            expect(page.locator('.output')).to_contain_text('earlier remote log')
            page.locator('#settings summary').click()
            page.locator('#style').select_option('colored')
            expect(page.locator('.output span')).to_have_text('Remote output')
            self.publisher.tick([dict(self.remote_task, pid_alive=False)])
            page.locator('#refresh-page').click()
            expect(page.locator('.card[data-task-id="run"]')).to_have_count(0)
            page.locator('#remote-tools summary').click()
            page.locator('#recent').check()
            expect(page.locator('.card[data-task-id="run"]')).to_contain_text('Process exited')
            expect(page.locator('.output')).to_contain_text('Remote output')
            page.locator('#settings summary').click()
            page.locator('#language').select_option('zh')
            expect(page.locator('h1')).to_have_text('任务中心')
            screenshot = os.environ.get('LEARNDL_TEST_SCREENSHOT')
            if screenshot:
                page.screenshot(path=screenshot)

            older = self.snapshot('older', 'complete', self.report)
            repository = self.repository
            repository.get_task.side_effect = {'old': self.finished, 'older': older}.get

            def finished_page(*, statuses, cursor, filters, **_):
                if 'complete' not in statuses:
                    return TaskPage((), 0, None, False)
                if filters.script or any(value is not None for value in (*filters.started, *filters.ended)):
                    rows = tuple(row for row in (self.finished, older) if filters.matches(row))
                    return TaskPage(rows, len(rows), None, False)
                if cursor:
                    return TaskPage((older,), 2, None, False)
                return TaskPage((self.finished,), 2, (self.finished.effective_start, 'old'), False)

            repository.list_tasks.side_effect = finished_page
            page.locator('nav a[href="/finished"]').click()
            card = page.locator('#sidebar').get_by_role('button', name='old.py')
            card.click()
            expect(card).to_have_attribute('aria-pressed', 'true')
            expect(page.locator('.live-output')).to_have_css('font-size', '18px')
            filters = page.locator('#finished-filters').bounding_box()
            sidebar = page.locator('#sidebar').bounding_box()
            detail = page.locator('.task-panel').bounding_box()
            assert filters is not None and sidebar is not None and detail is not None
            self.assertLessEqual(filters['y'] + filters['height'], sidebar['y'])
            self.assertLessEqual(sidebar['x'] + sidebar['width'], detail['x'])
            page.locator('#report summary').click()
            expect(page.frame_locator('#preview').locator('h1')).to_have_text('Report fixture')
            page.locator('#next-page').click()
            expect(page.locator('#sidebar').get_by_role('button', name='older.py')).to_be_visible()
            expect(page.locator('#task')).to_be_hidden()
            page.locator('#prev-page').click()
            expect(card).to_be_visible()
            page.locator('#filters summary').click()
            script = page.locator('#script-filter')
            script.fill('OLDER.PY')
            expect(page.locator('#sidebar').get_by_role('button', name='older.py')).to_be_visible()
            expect(card).to_have_count(0)
            script.fill('')
            expect(card).to_be_visible()
            start_range = page.locator('[data-date-range=started]')
            end_range = page.locator('[data-date-range=ended]')
            start_range.locator('[data-bound=from]').fill('2026-03-08')
            start_range.locator('[data-bound=before]').fill('2026-03-08')
            with page.expect_response(lambda response: '/api/finished?' in response.url and 'ended_before=' in response.url):
                end_range.locator('[data-bound=before]').fill('2026-03-09')
            zone = ZoneInfo('America/New_York')
            applied = repository.list_tasks.call_args.kwargs['filters']
            self.assertEqual(applied.started, (datetime(2026, 3, 8, tzinfo=zone).timestamp(),
                                               datetime(2026, 3, 9, tzinfo=zone).timestamp()))
            self.assertEqual(applied.ended, (None, datetime(2026, 3, 10, tzinfo=zone).timestamp()))
            start_range.locator('[data-bound=from]').fill('')
            start_range.locator('[data-bound=before]').fill('')
            end_range.locator('[data-bound=before]').fill('')
            expect(card).to_be_visible()
            page.locator('[name=status][value=complete]').uncheck()
            expect(card).to_have_count(0)
            expect(page.locator('#empty')).to_be_visible()
            self.assertEqual(errors, [])
            browser.close()
