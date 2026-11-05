"""Harmless runner smoke-test payload; deliberately contains no project imports."""
from __future__ import annotations

__all__ = []
import os

print(f'probe pid={os.getpid()} run={os.environ.get("LEARNDL_MANAGED_RUN")}', flush=True)
