"""Shared ordinary-task / exclusive maintenance admission, including nested calls."""
from __future__ import annotations

import contextvars
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import portalocker
import psutil

from .autofix_store import ROOT, Store, state_dir

_depth = contextvars.ContextVar('maintenance_depth', default=0)
_fd: contextvars.ContextVar[int | None] = contextvars.ContextVar('maintenance_fd', default=None)
_kind = contextvars.ContextVar('maintenance_kind', default='shared')


class MaintenanceBusy(RuntimeError):
    pass


def configured() -> bool:
    # The activation authority is outside the sandbox's writable project.
    directory = state_dir()
    if directory.exists():
        return Store(directory).enabled()
    from .config import installed_config
    config = installed_config() or {}
    if config.get('autofix_initialized'):
        raise RuntimeError('Autofix control state disappeared; refusing unguarded execution')
    return False


def sandbox_request():
    """Only present as a read-only mount inside the fixed autofix sandbox."""
    if not os.environ.get('LEARNDL_AUTOFIX_ID'):
        return None
    path = Path('/run/learndl-autofix.json')
    value = json.loads(path.read_text())
    if value['id'] != os.environ['LEARNDL_AUTOFIX_ID'] or value['nonce'] != os.environ.get('LEARNDL_AUTOFIX_NONCE'):
        raise RuntimeError('Invalid autofix execution context')
    return value


def inherited_fd(directory):
    try:
        if int(os.environ.get('LEARNDL_MAINTENANCE_OWNER', os.getpid())) == os.getpid():
            return None  # Another thread's environment is not an inherited lease.
        fd = int(os.environ['LEARNDL_MAINTENANCE_FD'])
        actual, expected = os.fstat(fd), (directory / 'gate.lock').stat()
        if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
            return fd
    except (KeyError, ValueError, OSError):
        pass
    return None


def pass_fds() -> tuple[int, ...]:
    fd = _fd.get()
    return () if fd is None else (fd,)


def child_env():
    fd = _fd.get()
    return {} if fd is None else {'LEARNDL_MAINTENANCE_FD': str(fd), 'LEARNDL_MAINTENANCE_OWNER': str(os.getpid())}


def pending(directory):
    return (directory / 'pending').exists() or bool(Store(directory).scheduled())


@contextmanager
def permit(*, exclusive=False, blocking=True, directory=None, priority=False, installation_id=None):
    if _depth.get() or sandbox_request() is not None:
        if exclusive and _kind.get() != 'exclusive':
            raise MaintenanceBusy('Cannot upgrade an existing shared lease')
        yield
        return
    if directory is None and not state_dir().exists() and not configured():
        yield
        return
    directory = directory or state_dir(ROOT)
    Store(directory)  # Missing/corrupt control state fails closed.
    inherited = inherited_fd(directory)
    if inherited is not None and not exclusive:
        token, fd_token = _depth.set(1), _fd.set(inherited)
        try:
            yield
        finally:
            _fd.reset(fd_token)
            _depth.reset(token)
        return
    stream = (directory / 'gate.lock').open('r+')
    waiter = directory / f'waiting-{os.getpid()}.json'
    previous = os.environ.get('LEARNDL_MAINTENANCE_FD')
    previous_owner = os.environ.get('LEARNDL_MAINTENANCE_OWNER')
    try:
        while True:
            try:
                with portalocker.Lock(str(directory / 'admission.lock'), mode='r+', timeout=0):
                    from .install_request import reconcile, ACTIVE
                    installation = reconcile()
                    if installation.get('phase') in ACTIVE - {'registering'} and installation.get('id') != installation_id:
                        raise MaintenanceBusy('An authorized installation is waiting for maintenance access')
                    if any(r['phase'] in {'starting', 'running'} for r in Store(directory).rows()):
                        raise MaintenanceBusy('Autofix execution or crash reconciliation is outstanding')
                    if not exclusive and not priority and pending(directory):
                        raise MaintenanceBusy('Autofix is waiting for maintenance access')
                    portalocker.lock(stream, (portalocker.LOCK_EX if exclusive else portalocker.LOCK_SH) | portalocker.LOCK_NB)
                break
            except (portalocker.LockException, MaintenanceBusy):
                if not blocking:
                    raise MaintenanceBusy('Maintenance execution lock unavailable') from None
                # Distinguish a parked legacy wrapper from unguarded live work.
                waiter.write_text(json.dumps({'pid': os.getpid(), 'created': psutil.Process().create_time()}))
                time.sleep(.5)
        waiter.unlink(missing_ok=True)
        token, fd_token = _depth.set(1), _fd.set(stream.fileno())
        kind_token = _kind.set('exclusive' if exclusive else 'shared')
        os.set_inheritable(stream.fileno(), True)
        os.environ.update(child_env())
        try:
            yield
        finally:
            _fd.reset(fd_token)
            _depth.reset(token)
            _kind.reset(kind_token)
            if previous_owner is None:
                os.environ.pop('LEARNDL_MAINTENANCE_OWNER', None)
            else:
                os.environ['LEARNDL_MAINTENANCE_OWNER'] = previous_owner
            if previous is None:
                os.environ.pop('LEARNDL_MAINTENANCE_FD', None)
            else:
                os.environ['LEARNDL_MAINTENANCE_FD'] = previous
            # flock belongs to the open file description. Explicit LOCK_UN
            # would also revoke children still holding an inherited descriptor.
            # Closing only this copy retains protection until the last child exits.
    finally:
        waiter.unlink(missing_ok=True)
        stream.close()
