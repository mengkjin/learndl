"""
Cross-platform terminal launcher package (cmux-first on macOS).

Public surface for embedding:

- :class:`Shell` — :meth:`~Shell.native` (background) and :meth:`~Shell.run` (visible terminal).
- :func:`format_python_command` — build a shell line to run a ``.py`` file.
- :class:`ProcessDiscovery` — find / wait for Python PIDs by script or ``TASK_ID`` (requires ``psutil``).

Platform-specific implementations live under ``terminal_opener.macos``, ``.windows``, and ``.linux``.
"""

from __future__ import annotations


from .util import format_python_command , ProcessDiscovery
from .shell import Shell

__all__ = [
    "Shell",
    "ProcessDiscovery",
    "format_python_command"
]
