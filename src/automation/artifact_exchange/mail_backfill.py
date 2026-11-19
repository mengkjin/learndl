"""Keep the original 0.7.2 patch import and preflight command working."""
from __future__ import annotations

from src.automation.version_patch import load_patch_support

__all__ = ['run', 'host_inputs']

_support = load_patch_support('0.7.2', 'mail_backfill')
run = _support.run
host_inputs = _support.host_inputs

if __name__ == '__main__':
    _support.main()
