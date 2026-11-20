"""Plain console logging before project initialization or code-lock admission.

Keep this module standard-library-only: maintenance commands must not initialize
``src.proj`` just to report their status. Resolve streams when emitting so output
redirection and task capture continue to work.
"""
from __future__ import annotations

import logging
import sys

__all__ = ['logger']


class _ConsoleHandler(logging.StreamHandler):
    def emit(self, record):
        use_stderr = record.levelno >= logging.ERROR or getattr(record, 'console_stderr', False)
        self.stream = sys.stderr if use_stderr else sys.stdout
        super().emit(record)


logger = logging.getLogger('learndl.automation.console')
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(_ConsoleHandler())
