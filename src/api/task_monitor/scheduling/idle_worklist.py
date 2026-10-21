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
from contextlib import ExitStack

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
    from .maintenance_gate import configured, pending
    from .autofix_store import state_dir
    if configured() and pending(state_dir()):
        return 'autofix maintenance pending'
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
    rerun_mark = WorklistState.rerun_mark(worklist)
    schedules, missing = {}, []
    for name in dict.fromkeys([*worklist['fit'], *WorklistState.requested_schedules()]):
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
                  'Other valid schedules remain eligible for idle training.\n'
                  f'Rerun mark: {rerun_mark}')
        # Notification-only event: never create a fictitious training run.
        store.event({'id': f'worklist-config-{key}', 'owner': OWNER}, 'configuration_error', detail)
    return worklist, schedules, missing


def candidate(store: RunStore, *, only: str | None = None, options: dict | None = None) -> dict | None:
    worklist, schedules, _ = worklist_schedules(store)
    return _candidate(store, worklist, schedules, only=only, options=options)


def _candidate(store: RunStore, worklist: dict, schedules: dict, *, only: str | None = None,
               options: dict | None = None) -> dict | None:
    from src.api.calls.worklist_state import WorklistState
    from .recovery import recovery_plan, record, evidence
    options = settings() if options is None else options
    resume, force = worklist.get('resume', False), worklist.get('force', False)
    attempts = store.owned(OWNER)
    # Explicit post-maintenance requests take precedence over ordinary worklist order.
    # Existing version/completion checks below still suppress repeat execution.
    ordered = sorted(schedules.items(), key=lambda item: 'training_request' not in item[1])
    for name, revisions in ordered:
        if only is not None and name != only:
            continue
        key = version(revisions)
        latest = next((run for run in reversed(attempts) if run['task'] == name and run['version'] == key
                       and run['phase'] != 'deferred'), None)
        recovery = {}
        if latest and latest['phase'] != 'complete' and not latest.get('retry_allowed'):
            if latest['phase'] in ACTIVE:
                continue
            recovery = recovery_plan(store, latest, options)
            previous_plan = evidence(store, latest['id']).get('recovery', {})
            if (previous_plan.get('recovery_status') in {'ready', 'cooldown', 'system memory busy', 'system memory unavailable'}
                    and recovery['recovery_status'] == 'cannot resume: recorded training directory missing'):
                store.event(latest, 'recovery_blocked', recovery['recovery_status'])
            record(store, latest['id'], recovery=recovery)
            if not recovery['retry']:
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
            resume_request = {field: recovery[field] for field in
                              ('parent_run_id', 'retry_index', 'resume_model_path', 'retry_not_before') if field in recovery}
            return {'task': name, 'version': key, 'revisions': revisions, **resume_request,
                    'rerun_mark': WorklistState.rerun_mark(worklist)}
    return None


def dispatch(options: dict, store: RunStore | None = None) -> dict:
    from . import rl_experiment as rl
    store = store or RunStore()
    with queue_lock():
        rl_request = rl.observe(store, throttle=True)
        # Superseded queues are deferrals, not attempted training. Running jobs
        # keep their exact snapshot even if the configuration changes.
        state = rl.status(store)
        for run in store.owned(OWNER):
            if run.get('task_kind') == rl.TASK and run['phase'] == 'queued':
                if state.get('version') != run['version'] or state.get('status') not in {'queued', 'pending'}:
                    run['phase'] = 'deferred'
                    store.put(run)
        worklist, schedules, missing = worklist_schedules(store) if options.get('enabled') else ({}, {}, [])
        diagnostics = {'missing_schedules': missing} if missing else {}
        if any(run['phase'] in ACTIVE for run in store.owned(OWNER)):
            rl.waiting(store, 'another idle training is queued or active')
            return {'idle_worklist': 'queued or active', **diagnostics}
        if reason := idle_reason(options):
            rl.waiting(store, reason)
            return {'idle_worklist': reason, **diagnostics}
        model_request = _candidate(store, worklist, schedules, options=options) if schedules else None
        last = next((r for r in reversed(store.owned(OWNER)) if r.get('started_at') and r['phase'] != 'deferred'), {})
        request = (rl_request if rl_request and (not model_request or last.get('task_kind') != rl.TASK)
                   else model_request)
        if request is None:
            return {'idle_worklist': 'no eligible schedules', **diagnostics}
        request.update(id=uuid.uuid4().hex, owner=OWNER, phase='queued', queued_at=time.time(),
                       progress_timeout_seconds=options.get('progress_timeout_seconds', 43200), timeout_seconds=None)
        store.put(request)
        return {'queued': request['task'], 'version': request['version'], **diagnostics}


def _launch(request: dict, store: RunStore) -> int:
    from .maintenance_gate import MaintenanceBusy, permit
    try:
        with permit(blocking=False):
            return _launch_admitted(request, store)
    except MaintenanceBusy:
        request['phase'] = 'deferred'
        store.put(request)
        return DEFERRED


def _launch_admitted(request: dict, store: RunStore) -> int:
    from src.proj import PATH
    log = runtime_dir() / 'idle_worklist' / f'{request["id"]}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    request.update(phase='starting', started_at=time.time(), uid=os.getuid(), log_path=str(log),
                   script_email_enabled=not request.get('no_email', False),
                   invocation_id=os.environ.get('INVOCATION_ID'),
                   runner_pid=os.getpid(), runner_created=psutil.Process().create_time(), boot_id=boot_id())
    store.put(request)
    read_fd, write_fd = os.pipe()
    child = None
    try:
        env = dict(os.environ, LEARNDL_MANAGED_RUN=request['id'], LEARNDL_GATE_FD=str(read_fd), PYTHONUNBUFFERED='1')
        from .maintenance_gate import pass_fds, child_env
        env.update(child_env())
        with log.open('ab', buffering=0) as output:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'src.api.task_monitor.scheduling.idle_worklist', '_child', request['id']],
                                     cwd=PATH.main, env=env, start_new_session=True, pass_fds=(read_fd, *pass_fds()),
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
                if request.get('task_kind') == 'rl_experiment':
                    from .rl_experiment import observe
                    next_candidate = observe(store)
                else:
                    next_candidate = candidate(store, only=request['task'], options=options) if options.get('enabled') else None
                if (others or idle_reason(options) or not next_candidate or next_candidate['version'] != request['version']
                        or next_candidate.get('parent_run_id') != request.get('parent_run_id')
                        or next_candidate.get('retry_index') != request.get('retry_index')):
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
    store = RunStore()
    request = store.get(run_id)
    if request.get('task_kind') == 'rl_experiment':
        from .rl_experiment import execute
        from .recovery import record
        try:
            return execute(request, store)
        except BaseException as exc:
            record(store, run_id, script_outcome='error', error=f'{type(exc).__name__}: {exc}')
            raise
    from src.api.calls.research import CarryOutScheduleWorkList
    from src.api.calls.worklist_state import AutomaticDeferred
    from .recovery import record
    sys.argv = [str(CarryOutScheduleWorkList.SCHEDULE_SCRIPT)]
    try:
        CarryOutScheduleWorkList(only_schedule=request['task'], automatic=True,
                                 expected_version=request['version'],
                                 recovery_run_id=run_id if request.get('parent_run_id') else None).run()
    except AutomaticDeferred:
        return DEFERRED
    except BaseException as exc:
        record(store, run_id, script_outcome='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'error')
        raise
    record(store, run_id, script_outcome='success')
    return 0


def reset_worklist(store: RunStore | None = None) -> str:
    """Treat the current worklist as new locally, preserving YAML and history.

    Training starts on subsequent idle polls and retains the worklist resume policy.
    Refuse queued/active automatic work or a locked worklist schedule.
    """
    from src.proj import PATH
    from src.api.calls.worklist_state import WorklistState
    from src.res.model.util.training_history import now, write_json
    store = store or RunStore()
    with queue_lock(), ExitStack() as locks:
        if any(run['phase'] in ACTIVE for run in store.owned(OWNER)):
            raise RuntimeError('Cannot reset a queued or active worklist')
        _, schedules, missing = worklist_schedules(store)
        if missing:
            raise ValueError(f'Cannot reset worklist with missing schedules: {", ".join(missing)}')
        try:
            for name in sorted(schedules):
                locks.enter_context(WorklistState(name).lock())
        except portalocker.LockException as exc:
            raise RuntimeError('Cannot reset an active worklist schedule') from exc
        generation = uuid.uuid4().hex
        write_json(PATH.lc_machine / 'schedule_worklist' / 'reset.json',
                   {'generation': generation, 'reset_at': now()})
        return generation


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
    parser.add_argument('command', choices=['worker', '_child', 'status', 'rl-status', 'reset', 'reset-worklist'])
    parser.add_argument('schedule_or_run', nargs='?')
    args = parser.parse_args()
    if args.command == 'worker':
        return worker()
    if args.command == 'reset-worklist':
        print(f'Worklist reset: {reset_worklist()}')
        return 0
    if args.command in {'status', 'rl-status'}:
        from .recovery import evidence
        store = RunStore()
        runs = [dict(run, evidence=evidence(store, run['id'])) for run in store.owned(OWNER)]
        if args.command == 'rl-status':
            from .rl_experiment import status
            result = {'configuration': status(store),
                      'runs': [run for run in runs if run.get('task_kind') == 'rl_experiment']}
        else:
            result = runs  # Preserve the existing status JSON list.
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if not args.schedule_or_run:
        parser.error('schedule/run ID required')
    if args.command == 'reset':
        reset(args.schedule_or_run)
        return 0
    return child(args.schedule_or_run)


if __name__ == '__main__':
    raise SystemExit(main())
