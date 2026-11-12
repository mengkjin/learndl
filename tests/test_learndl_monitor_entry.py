"""Check fresh-browser console entry and same-origin authentication boundaries."""
from __future__ import annotations

from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.api.learndl_monitor import __main__ as console

__all__ = []


class ConsoleEntryTest(unittest.TestCase):
    def test_root_bootstraps_current_token_but_api_still_requires_it(self):
        handlers = []

        class Server:
            def __init__(self, address, handler):
                handlers.append(handler)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def serve_forever(self):
                pass

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(console, 'STATE', Path(temporary)), \
                patch.object(console, 'ThreadingHTTPServer', Server), \
                patch.object(console, 'Manager'), \
                patch.object(console.secrets, 'token_urlsafe', return_value='fresh-token'), \
                patch('sys.argv', ['monitor', '--serve']):
            console.main()

        def request(path, headers):
            handler = object.__new__(handlers[0])
            handler.path = path
            handler.headers = headers
            handler.wfile = BytesIO()
            handler.send_response = MagicMock()
            handler.send_header = MagicMock()
            handler.end_headers = MagicMock()
            handler.send_error = MagicMock()
            handler.do_GET()
            return handler

        for path in ('/', '/?review=1'):
            page = request(path, {'Host': '127.0.0.1:8500'})
            page.send_response.assert_called_once_with(200)
            self.assertIn(b'content="fresh-token"', page.wfile.getvalue())
            page.send_header.assert_any_call('Cache-Control', 'no-store')
        wrong_host = request('/', {'Host': 'attacker.example:8500'})
        wrong_host.send_error.assert_called_once_with(403)
        wrong_origin = request('/', {'Host': '127.0.0.1:8500', 'Origin': 'http://attacker.example'})
        wrong_origin.send_error.assert_called_once_with(403)
        missing_token = request('/api/services', {'Host': '127.0.0.1:8500'})
        missing_token.send_response.assert_called_once_with(403)
        self.assertIn(b'Session expired', missing_token.wfile.getvalue())


if __name__ == '__main__':
    unittest.main()
