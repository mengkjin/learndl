"""Helpers for running pipeline scripts from a DirectCall ``python -c`` session."""
from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ['as_script_main']


@contextmanager
def as_script_main(script_path: Path) -> Iterator[None]:
    """Point ``__main__.__file__`` at *script_path* for ScriptTool task recording."""
    main_module = sys.modules['__main__']
    previous_file = getattr(main_module, '__file__', None)
    main_module.__file__ = str(script_path.resolve())
    try:
        yield
    finally:
        main_module.__file__ = previous_file
