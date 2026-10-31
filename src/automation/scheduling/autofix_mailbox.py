"""Host access to an untrusted sandbox mailbox, without following symlinks."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import stat


@contextmanager
def directory(path: Path, *, create=False):
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Mailbox path must be absolute and canonical')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def create(path):
    with directory(path, create=True):
        pass


def write(path, name, value):
    """Pinned parent + exclusive temporary file + atomic replacement of the link itself."""
    with directory(path) as fd:
        temporary = '.host-' + secrets.token_hex(16)
        output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        try:
            with os.fdopen(output, 'w') as stream:
                json.dump(value, stream)
            os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=fd)
            except FileNotFoundError:
                pass


def read(path, name):
    with directory(path) as fd:
        source = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(source, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError('Mailbox events must be regular files')
            raw = stream.read(65537)
            if len(raw) > 65536:
                raise ValueError('Mailbox event exceeds size limit')
            return json.loads(raw)
