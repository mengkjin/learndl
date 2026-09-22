"""Queue idle work in watchdog; execute one schedule in an independent service."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid

import portalocker
import psutil
import yaml

from .config import installed_config, runtime_dir
from .runtime import RunStore, members, process_cgroup, boot_id

OWNER = 'idle_worklist'
ACTIVE = {'queued', 'starting', 'running', 'terminating', 'finalizing'}
DEFERRED = 75


def settings() -> dict:
    config = installed_config() or {}
    return config.get('watchdog', {}).get('jobs', {}).get(OWNER, {})


def queue_lock():
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return portalocker.Lock(str(directory / 'idle_queue.lock'), timeout=0)


def version(revisions: dict) -> str:
    return hashlib.sha256(json.dumps(revisions, sort_keys=True).encode()).hexdigest()


def gpu_memory_percent() -> float:
    """Probe the GPU used as cuda:0; an ambiguous/unreadable device is not idle."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            timeout=10, check=True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
    rows = [tuple(value.strip() for value in line.split(',')) for line in result.stdout.splitlines()]
    selected = [row for row in rows if len(row) == 4 and
                (row[0] == visible or (visible.startswith('GPU-') and row[1].startswith(visible)))]
    if len(selected) != 1 or float(selected[0][3]) <= 0:
        raise ValueError('Cannot identify cuda:0 GPU memory')
    return float(selected[0][2]) / float(selected[0][3]) * 100


def idle_reason(options: dict) -> str | None:
    from src.proj import MACHINE
    from src.proj.util.script import FitLock
    if not MACHINE.cuda_server or sys.platform != 'linux':
        return 'requires a CUDA server on Linux'
    if FitLock.is_held():
        return 'fit lock occupied'
    try:
        used = gpu_memory_percent()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return f'GPU probe unavailable: {exc}'
    if not 0 <= used < options.get('max_gpu_memory_percent', 20):
        return f'GPU memory busy: {used:.1f}%'
    return None


def worklist_schedules(store: RunStore) -> tuple[dict, dict, list[str]]:
    """Validate every name and durably report missing configs, even while busy."""
    from src.proj import MACHINE, PATH
    from src.api.calls.worklist_state import WorklistState
    from src.res.model.util.training_history import file_revision
    worklist_revision = file_revision(PATH.sched_worklist)
    worklist = yaml.safe_load(worklist_revision['content'])
    resume, force = worklist.get('resume', False), worklist.get('force', False)
    if type(resume) is not bool or type(force) is not bool:
        raise ValueError('worklist resume and force must be YAML booleans')
    schedules, missing = {}, []
    for name in dict.fromkeys(worklist['fit']):
        try:
            revisions = WorklistState.revisions(name)
        except FileNotFoundError:
            missing.append(name)
            continue
        revisions['worklist'] = worklist_revision
        schedules[name] = revisions
    if missing:
        key = version({'worklist': worklist_revision, 'missing': sorted(missing)})
        detail = (f'Host: {MACHINE.name}\nWorklist: {PATH.sched_worklist}\n'
                  f'Git commit: {worklist_revision["git_commit"]}\n'
                  f'Worklist SHA256: {worklist_revision["sha256"]}\n'
                  f'Missing schedule configs: {", ".join(missing)}\n'
                  f'Searched directories:\n{PATH.sched}\n{PATH.sched_shared}\n'
                  'Worklist entries must match schedule filenames without .yaml, '
                  'not model modules or input data types.\n'
                  'These schedules cannot start, even with force: true. '
                  'Correct the names or add the missing config files. '
                  'Other valid schedules remain eligible for idle training.')
        # Notification-only event: never create a fictitious training run.
        store.event({'id': f'worklist-config-{key}', 'owner': OWNER}, 'configuration_error', detail)
    return worklist, schedules, missing


def candidate(store: RunStore, *, only: str | None = None) -> dict | None:
    worklist, schedules, _ = worklist_schedules(store)
    return _candidate(store, worklist, schedules, only=only)


def _candidate(store: RunStore, worklist: dict, schedules: dict, *, only: str | None = None) -> dict | None:
    from src.api.calls.worklist_state import WorklistState
    resume, force = worklist.get('resume', False), worklist.get('force', False)
    attempts = store.owned(OWNER)
    for name, revisions in schedules.items():
        if only is not None and name != only:
            continue
        key = version(revisions)
        if any(run['task'] == name and run['version'] == key and
               run['phase'] not in {'deferred', 'complete'} and not run.get('retry_allowed')
               for run in attempts):
            continue
        state = WorklistState(name)
        try:
            with state.lock():
                previous = state.read()
                already_completed = previous and previous.get('status') == 'success' and previous.get('revisions') == revisions
                reason, _ = state.decision(previous, revisions, force=force and not already_completed, resume=resume)
        except portalocker.LockException:
            continue
        if reason != 'completed':
            return {'task': name, 'version': key, 'revisions': revisions}
    return None


def dispatch(options: dict, store: RunStore | None = None) -> dict:
    store = store or RunStore()
    with queue_lock():
        worklist, schedules, missing = worklist_schedules(store)
        diagnostics = {'missing_schedules': missing} if missing else {}
        if any(run['phase'] in ACTIVE for run in store.owned(OWNER)):
            return {'idle_worklist': 'queued or active', **diagnostics}
        if reason := idle_reason(options):
            return {'idle_worklist': reason, **diagnostics}
        request = _candidate(store, worklist, schedules)
        if request is None:
            return {'idle_worklist': 'no eligible schedules', **diagnostics}
        request.update(id=uuid.uuid4().hex, owner=OWNER, phase='queued', queued_at=time.time(),
                       progress_timeout_seconds=options.get('progress_timeout_seconds', 43200), timeout_seconds=None)
        store.put(request)
        return {'queued': request['task'], 'version': request['version'], **diagnostics}


def _launch(request: dict, store: RunStore) -> int:
    from src.proj import PATH
    log = runtime_dir() / 'idle_worklist' / f'{request["id"]}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    request.update(phase='starting', started_at=time.time(), uid=os.getuid(), log_path=str(log),
                   script_email_enabled=True,
                   runner_pid=os.getpid(), runner_created=psutil.Process().create_time(), boot_id=boot_id())
    store.put(request)
    read_fd, write_fd = os.pipe()
    child = None
    try:
        env = dict(os.environ, LEARNDL_MANAGED_RUN=request['id'], LEARNDL_GATE_FD=str(read_fd), PYTHONUNBUFFERED='1')
        with log.open('ab', buffering=0) as output:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'src.api.task_monitor.scheduling.idle_worklist', '_child', request['id']],
                                     cwd=PATH.main, env=env, start_new_session=True, pass_fds=(read_fd,),
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        request.update(pid=child.pid, process_created=psutil.Process(child.pid).create_time(), phase='running')
        cgroup = process_cgroup(child.pid)
        if cgroup and cgroup.endswith('/learndl-idle-worklist.service'):
            request['cgroup'] = cgroup
        store.put(request)
        store.event(request, 'started', 'Automatic schedule process started.')
        os.write(write_fd, b'1')
    except BaseException as exc:
        if child is not None:
            child.kill()
            child.wait()
        request.update(phase='error', finished_at=time.time())
        store.put(request)
        store.event(request, 'finished', f'Launcher failed: {type(exc).__name__}: {exc}')
        raise
    finally:
        os.close(read_fd)
        os.close(write_fd)
    code = child.wait()
    while members(request):
        time.sleep(1)
    with store.connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        current = store.get(request['id'])
        current['returncode'] = code
        connection.execute('UPDATE runs SET data=? WHERE id=?', (json.dumps(current), request['id']))
    # Watchdog finalizes and sends durable notifications, including if we die here.
    return code if code >= 0 else 128 - code


def worker() -> int:
    """Minutely independent trigger; a queue item is required to do any work."""
    from src.proj.util.script.script_lock import ScriptLock
    try:
        with ScriptLock('idle_worklist_worker', timeout=1, wait_time=2):
            store = RunStore()
            with queue_lock():
                queued = [run for run in store.owned(OWNER) if run['phase'] == 'queued']
                if not queued:
                    return 0
                request = queued[0]
                options = settings()
                others = [run for run in store.owned(OWNER) if run['phase'] in ACTIVE and run['id'] != request['id']]
                # Candidate must ignore its own queued row, while preserving failed attempts.
                request['phase'] = 'deferred'
                store.put(request)
                next_candidate = candidate(store, only=request['task']) if options.get('enabled') else None
                if others or idle_reason(options) or not next_candidate or next_candidate['version'] != request['version']:
                    return 0
                # Claim durably before releasing the short queue lock.
                request.update(phase='starting', started_at=time.time())
                store.put(request)
            return _launch(request, store)
    except (portalocker.LockException, BlockingIOError, TimeoutError):
        return 0


def child(run_id: str) -> int:
    gate = int(os.environ.pop('LEARNDL_GATE_FD'))
    ready = os.read(gate, 1)
    os.close(gate)
    if ready != b'1':
        return 1
    request = RunStore().get(run_id)
    from src.api.calls.research import CarryOutScheduleWorkList
    from src.api.calls.worklist_state import AutomaticDeferred
    sys.argv = [str(CarryOutScheduleWorkList.SCHEDULE_SCRIPT)]
    try:
        CarryOutScheduleWorkList(only_schedule=request['task'], automatic=True,
                                 expected_version=request['version']).run()
    except AutomaticDeferred:
        return DEFERRED
    return 0


def reset(schedule: str, store: RunStore | None = None) -> None:
    store = store or RunStore()
    with queue_lock():
        runs = [run for run in store.owned(OWNER) if run['task'] == schedule]
        if any(run['phase'] in ACTIVE for run in runs):
            raise RuntimeError('Cannot reset a queued or active schedule')
        for run in runs:
            run['retry_allowed'] = True
            store.put(run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['worker', '_child', 'status', 'reset'])
    parser.add_argument('schedule_or_run', nargs='?')
    args = parser.parse_args()
    if args.command == 'worker':
        return worker()
    if args.command == 'status':
        print(json.dumps(RunStore().owned(OWNER), indent=2, ensure_ascii=False))
        return 0
    if not args.schedule_or_run:
        parser.error('schedule/run ID required')
    if args.command == 'reset':
        reset(args.schedule_or_run)
        return 0
    return child(args.schedule_or_run)


if __name__ == '__main__':
    raise SystemExit(main())
