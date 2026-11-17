"""Bootstrap logging stays plain, redirectable and independent of project setup."""
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

from src.automation.console import logger


class ConsoleTest(unittest.TestCase):
    def test_redirected_streams_and_json(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            logger.info(json.dumps({'status': 'ready', 'progress': '50%'}))
            logger.error('failed: %s', 'example')
        self.assertEqual(json.loads(out.getvalue()), {'status': 'ready', 'progress': '50%'})
        self.assertEqual(err.getvalue(), 'failed: example\n')
        second = io.StringIO()
        with redirect_stdout(second):
            logger.info('next capture')
        self.assertEqual(second.getvalue(), 'next capture\n')

    def test_bootstrap_does_not_import_project(self):
        result = subprocess.run(
            [sys.executable, '-c',
             'import sys; from src.automation.console import logger; '
             'assert "src.proj" not in sys.modules; logger.info("ready")'],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout, 'ready\n')
        self.assertEqual(result.stderr, '')
