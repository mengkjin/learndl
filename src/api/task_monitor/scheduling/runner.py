"""Run only installed project entrypoints with shared locks and durable identity."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from .config import ENTRYPOINTS, installed_config
from .runtime import RunStore, members


def run(task_id: str) -> int:
    from src.proj import PATH
    from src.proj.util.script.script_lock import ScriptLockMultiple
    config = installed_config()
    if config is None:
        raise RuntimeError('No installed scheduling snapshot; run installer apply first')
    task = config['tasks'][task_id]
    if not task['enabled']:
        raise ValueError('Task is disabled')
    lock = ScriptLockMultiple(task['lock_group'], 1,
                              timeout=task['wait_timeout_seconds'] if task['overlap'] == 'wait' else 1,
                              wait_time=1 if task['overlap'] == 'wait' else 2)
    try:
        lock.__enter__()
    except (BlockingIOError, TimeoutError):
        print(f'SCHEDULER SKIPPED: {task_id}: execution lock unavailable', flush=True)
        return 0
    try:
        store = RunStore()
        # A previous runner can die before its descendants release inherited
        # descriptors. Its durable session identity remains authoritative even
        # when no file lock survives in a deliberately detached child.
        wait_started = time.monotonic()
        for older in store.active():
            if older.get('lock_group') != task['lock_group'] or older['phase'] == 'starting':
                continue
            while members(older):
                if task['overlap'] == 'skip' or time.monotonic() - wait_started >= task['wait_timeout_seconds']:
                    print(f'SCHEDULER SKIPPED: {task_id}: previous supervised session remains active', flush=True)
                    return 0
                time.sleep(1)
        run_id = uuid.uuid4().hex
        run_record = {'id': run_id, 'task': task_id, 'phase': 'starting', 'started_at': time.time(),
                      'timeout_seconds': task['timeout_seconds'], 'uid': os.getuid(), 'lock_group': task['lock_group']}
        store.put(run_record)
        read_fd, write_fd = os.pipe()
        lock_fd = lock.acquired_locks[0].fileno()
        env = os.environ.copy()
        env.update(LEARNDL_MANAGED_RUN=run_id, LEARNDL_LOCK_FD=str(lock_fd),
                   LEARNDL_LOCK_GROUP=task['lock_group'], LEARNDL_GATE_FD=str(read_fd))
        script = str(PATH.main / ENTRYPOINTS[task['entrypoint']])
        command = [sys.executable, '-m', 'src.api.task_monitor.scheduling.runner', '_child', script, *task['args']]
        child = subprocess.Popen(command, cwd=PATH.main, env=env, start_new_session=True, pass_fds=(read_fd, lock_fd))
        os.close(read_fd)
        try:
            from .runtime import process_cgroup
            run_record.update(pid=child.pid, runner_pid=os.getpid(), runner_created=psutil.Process().create_time(), process_created=psutil.Process(child.pid).create_time(), phase='running', started_at=time.time())
            cgroup = process_cgroup(child.pid)
            if cgroup and cgroup.endswith(f'/learndl-schedule-{task_id}.service'):
                run_record['cgroup'] = cgroup
            store.put(run_record)
            os.write(write_fd, b'1')  # Child may only execute after durable registration.
        finally:
            os.close(write_fd)
        returncode = child.wait()
        while members(run_record):
            time.sleep(1)  # Keep the execution lock while descendants remain.
        # Watchdog is the single finalizer: do not overwrite termination intent.
        with store.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            current = store.get(run_id)
            current['returncode'] = returncode
            import json
            connection.execute('UPDATE runs SET data=? WHERE id=?', (json.dumps(current), run_id))
        return returncode if returncode >= 0 else 128 - returncode
    finally:
        lock.__exit__(None, None, None)


def inherited_lock(lock_group: str) -> bool:
    """Only the registered session leader may bypass its inherited outer lock."""
    from src.proj.util.script.script_lock import ScriptLockMultiple
    try:
        if os.environ.get('LEARNDL_LOCK_GROUP') != lock_group:
            return False
        fd = int(os.environ['LEARNDL_LOCK_FD'])
        expected = ScriptLockMultiple.LOCK_DIR / lock_group / 'instance.0.lock'
        actual = os.fstat(fd)
        target = expected.stat()
        run_record = RunStore().get(os.environ['LEARNDL_MANAGED_RUN'])
        return (actual.st_dev, actual.st_ino) == (target.st_dev, target.st_ino) and run_record.get('pid') == os.getpid()
    except (KeyError, ValueError, OSError):
        return False


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == '_child':
        gate = int(os.environ.pop('LEARNDL_GATE_FD'))
        ready = os.read(gate, 1)
        os.close(gate)
        if ready != b'1':
            return 1
        os.set_inheritable(int(os.environ['LEARNDL_LOCK_FD']), True)
        os.execv(sys.executable, [sys.executable, str(Path(sys.argv[2])), '--source=bash', '--email=1', *sys.argv[3:]])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run'])
    parser.add_argument('task')
    args = parser.parse_args()
    return run(args.task)


if __name__ == '__main__':
    raise SystemExit(main())
