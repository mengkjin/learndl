"""Compatibility entry point for previously installed version patch launchers."""
from __future__ import annotations

import sys

from src.automation import version_patch as _implementation
from src.automation.version_patch import __all__ as __all__


if __name__ == '__main__':
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
