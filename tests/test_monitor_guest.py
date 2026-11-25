"""Guest checks at the actual service entry points, before mutations."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, call, patch

from src.api.learndl_monitor import __main__ as console, experiments, library_server


class GuestAccessTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name)

    def handler(self, module, port, launch):
        with patch.object(module, 'ThreadingHTTPServer') as server, \
                patch.object(module.secrets, 'token_urlsafe', return_value='test-token'):
            launch()
        implementation = server.call_args.args[1]

        def request(path, data=None, *, guest=True):
            origin = f'127.0.0.1:{port}'
            body = json.dumps(data).encode() if data is not None else b''
            handler = object.__new__(implementation)
            handler.path = path
            handler.headers = {'Host': origin, 'Origin': 'http://' + origin,
                               'Authorization': 'Bearer test-token', 'Content-Length': str(len(body))}
            if guest:
                handler.headers.update({'X-LearnDL-Role': 'admin', 'X-Forwarded-Proto': 'https'})
            handler.rfile, handler.wfile = BytesIO(body), BytesIO()
            handler.send_response, handler.send_header, handler.end_headers = MagicMock(), MagicMock(), MagicMock()
            if data is None:
                handler.do_GET()
            else:
                handler.do_POST()
            return handler.send_response.call_args.args[0], handler.wfile.getvalue()

        return request

    def test_console_guest_can_control_viewers(self):
        with patch.object(console, 'STATE', self.state), patch.object(console, 'Manager') as manager, \
                patch('sys.argv', ['monitor', '--serve']):
            request = self.handler(console, 8500, console.main)
        for action in ('start', 'stop', 'restart'):
            self.assertEqual(request('/api/action', {'service': 'tasks', 'action': action})[0], 202)
        self.assertEqual(manager.return_value.submit.call_args_list,
                         [call('tasks', action, None, None) for action in ('start', 'stop', 'restart')])
        self.assertEqual(request('/api/preferences', {'show_all': True})[0], 200)
        self.assertEqual(request('/api/action', {'service': 'research', 'action': 'start'})[0], 403)
        manager.return_value.restart_running.return_value = {'active': True}
        targets = [{'id': 'tasks', 'pid': 1, 'created': 1.0}]
        self.assertEqual(request('/api/restart-services', {'targets': targets})[0], 202)
        mixed_targets = targets + [{'id': 'research', 'pid': 2, 'created': 2.0}]
        self.assertEqual(request('/api/restart-services', {'targets': mixed_targets})[0], 403)
        manager.return_value.restart_running.assert_called_once_with(targets)
        self.assertEqual(request('/api/action', {'service': 'research', 'action': 'start'}, guest=False)[0], 202)
        guest_page = request('/')[1]
        admin_page = request('/', guest=False)[1]
        self.assertIn(b'data-monitor-role="guest"', guest_page)
        self.assertIn(b'data-monitor-role="admin"', admin_page)
        self.assertIn(b'id="syncthing-link" role="link" aria-disabled="true"', guest_page)
        self.assertNotIn(b'http://127.0.0.1:8384/', guest_page)
        self.assertIn(b'id="syncthing-link" href="http://127.0.0.1:8384/" target="_blank"', admin_page)

    def test_guest_can_preview_models_and_manage_mail_without_deleting_data(self):
        with patch.object(library_server, 'Library') as library:
            model = self.handler(library_server, 8507, lambda: library_server.main('model_preview', state=self.state))
            mail = self.handler(library_server, 8506, lambda: library_server.main('mail_archive', state=self.state))
        catalog = library.return_value
        catalog.model_listing.return_value = [{'name': 'test model'}]
        catalog.mail_headers.return_value = {'From': 'fixture@example.com'}
        catalog.manage.return_value = []
        self.assertIn(b'test model', model('/api/models')[1])
        self.assertIn(b'fixture@example.com', mail('/api/mail-headers?id=test')[1])
        self.assertEqual(model('/preview')[0], 200)
        self.assertEqual(model('/manage')[0], 403)
        self.assertEqual(model('/api/model-management')[0], 403)
        self.assertEqual(model('/manage', guest=False)[0], 200)
        with patch('src.res.model.workflow.model_management.manage', return_value=[]) as manage:
            operation = {'action': 'apply_models', 'operation': 'rename', 'items': [{'id': 'test'}]}
            for action in ('plan_models', 'apply_models', 'history_models', 'undo_models'):
                self.assertEqual(model('/api/action', dict(operation, action=action))[0], 403)
            manage.assert_not_called()
            self.assertEqual(model('/api/action', operation, guest=False)[0], 200)
            manage.assert_called_once()
        for action in ('delete', 'purge'):
            self.assertEqual(mail('/api/action', {'action': action, 'ids': ['test']})[0], 403)
        catalog.manage.assert_not_called()
        for action in ('archive', 'unarchive', 'restore'):
            self.assertEqual(mail('/api/action', {'action': action, 'ids': ['test']})[0], 200)
        self.assertEqual(mail('/api/action', {'action': 'delete', 'ids': ['test']}, guest=False)[0], 200)
        self.assertEqual(catalog.manage.call_args_list,
                         [call(action, ['test']) for action in ('archive', 'unarchive', 'restore', 'delete')])

    def test_guest_can_only_start_a_listed_experiment(self):
        manager = MagicMock()
        manager.state = self.state
        manager.sources.return_value = {'optuna': [{'items': [{'path': '/listed.db'}]}]}
        manager.act.return_value = {'pid': 123}
        with patch.object(experiments, 'Instances', return_value=manager):
            request = self.handler(experiments, 8505, experiments.main)
        start = {'action': 'start', 'kind': 'optuna', 'source': '/listed.db'}
        self.assertEqual(request('/api/action', start)[0], 200)
        self.assertEqual(request('/api/action', dict(start, source='/unlisted.db'))[0], 403)
        stop = {'action': 'stop', 'kind': 'optuna', 'pid': 123, 'created': 1}
        self.assertEqual(request('/api/action', stop)[0], 200)
        self.assertEqual(manager.act.call_args_list, [call(start, guest=True), call(stop, guest=True)])
