"""Serialize cooperating writers of one filesystem output."""
import os
from contextlib import contextmanager
from pathlib import Path

import portalocker


@contextmanager
def output_lock(path: Path):
    sidecar = path.with_name(f'.{path.name}.write-lock')
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with sidecar.open('a+b') as handle:
            portalocker.lock(handle, portalocker.LOCK_EX)
            # A previous owner can unlink the sidecar while this opener waits.
            # Only the owner of the currently linked inode may enter.
            if os.fstat(handle.fileno()).st_nlink == 0:
                continue
            try:
                yield
            finally:
                # Windows requires a stable sidecar while other handles are open.
                if os.name == 'posix':
                    sidecar.unlink()
            return
