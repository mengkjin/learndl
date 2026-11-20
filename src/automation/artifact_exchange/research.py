"""Explicit research result adapters. Failed delivery never masks research outcomes."""
from __future__ import annotations
import contextvars
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import replace
import uuid
from typing import Any
import json
import logging
from pathlib import Path
from .core import ArtifactSpec, Config, publish_artifacts, sandboxed

__all__ = [
    'ANALYSIS_SUFFIXES',
    'EXCLUDED_DIRS',
    'TaskDelivery',
    'task_delivery',
    'delivery_log',
    'log_results',
    'publish_safe',
    'finalize_task_artifacts',
    'defer_mail',
    'run_deferred_mail',
    'register',
    'analysis_specs',
    'publish_analysis',
    'delivered',
    'notice',
    'handoff_bundle',
    'bundle_delivery',
    'register_report',
    'model_run',
    'delivered_report_paths',
]

MODEL_ARTIFACTS: contextvars.ContextVar[list[ArtifactSpec] | None] = contextvars.ContextVar('artifact_model', default=None)
CURRENT_TASK: contextvars.ContextVar[Any] = contextvars.ContextVar('artifact_task', default=None)
ANALYSIS_SUFFIXES = {'.pdf', '.xlsx', '.json', '.csv', '.html', '.md', '.txt', '.log'}
EXCLUDED_DIRS = {'models', 'model', 'checkpoints', 'checkpoint', 'snapshots', 'data', 'dataset', 'prepared', 'mail', 'training_mail', 'bundles', 'tensorboard', 'recovery'}


class TaskDelivery:
    def __init__(self):
        self.pending = []
        self.hooks = []
        self.bundles = {}
        self.mail = []
        self.flushing = False
        self.mailing = False


def task_delivery(owner=None):
    owner = owner if owner is not None else CURRENT_TASK.get()
    if owner is None:
        return None
    if getattr(owner, '_artifact_delivery', None) is None:
        owner._artifact_delivery = TaskDelivery()
    return owner._artifact_delivery


def delivery_log(message, *, failed=False, owner=None):
    owner = owner if owner is not None else CURRENT_TASK.get()
    logger = getattr(owner, 'logger', None)
    if logger is not None:
        (logger.warning if failed else logger.note)('[artifacts] ' + message, vb_level=0)
    else:
        logger = logging.getLogger(__name__)
        (logger.warning if failed else logger.info)('[artifacts] ' + message)


def _visible_result(record):
    """Routine opt-out is quiet; failed work and blocked prepared packages remain visible."""
    reason = str(record.get('reason', ''))
    return not (record.get('status') == 'skipped' and not record.get('id') and
                (reason in {'autofix', 'publisher disabled', 'publisher disabled or autofix'} or reason == 'model files unchanged' or
                 reason.startswith('publisher inactive:')))


def log_results(records, owner=None):
    # Task-owned deliveries are reported once at finalization, inside capture.
    # Keep package details in the delivery records/outbox rather than every email.
    owner = owner if owner is not None else CURRENT_TASK.get()
    if owner is not None:
        return
    for record in records:
        if not _visible_result(record):
            continue
        manifest = record.get('manifest', {})
        text = f"{record.get('status', 'unknown')} id={record.get('id', '-')}"
        if manifest:
            text += f" experiment={manifest.get('category')}/{manifest.get('experiment')} run={manifest.get('run_id')}"
        if record.get('route'):
            text += f" destination=artifact_exchange/{record['route']}/{record['filename']}"
        if record.get('error') or record.get('reason'):
            text += ' reason=' + str(record.get('error') or record.get('reason'))
        delivery_log(text, failed=record.get('status') in ('failed', 'skipped'), owner=owner)


def publish_safe(task, run_id, artifacts, status='success', *, build=None):
    owner = CURRENT_TASK.get()
    queue = task_delivery(owner)
    config = None
    try:
        if sandboxed():
            records = [{'status': 'skipped', 'reason': 'autofix'}]
        else:
            config = Config.load()
            records = publish_artifacts(task, run_id, artifacts, status, config=config,
                                        prepare_only=queue is not None and not queue.flushing, build=build)
    except Exception as exc:
        records = [{'status': 'failed', 'error': str(exc)}]
    fresh = records
    if owner is not None:
        known = {r['id']: r for r in owner._artifact_results if r.get('id')}
        fresh = [r for r in records if not r.get('id') or r['id'] not in known]
        records = [known.get(r.get('id'), r) for r in records]
        owner._artifact_results.extend(fresh)
    if queue is not None and not queue.flushing:
        if fresh:
            queue.pending.append((config, fresh))
    else:
        log_results(fresh, owner)
    return records


def finalize_task_artifacts(owner, saves_complete=True):
    """Run while the task's HTML/Markdown capture is still open, before any mail."""
    from .core import send_prepared
    queue = task_delivery(owner)
    assert queue is not None
    queue.flushing = True
    if owner._artifacts:
        if saves_complete:
            publish_safe(owner.task_name, getattr(owner, 'task_run_id', None) or owner.init_time.isoformat(),
                         owner._artifacts, owner.execution_status)
        else:
            record = {'status': 'skipped', 'reason': 'asynchronous saves incomplete'}
            owner._artifact_results.append(record)
            log_results([record], owner)
    for config, records in queue.pending:
        try:
            if config is not None:
                send_prepared(records, config)
        except Exception as exc:
            for record in records:
                if record.get('status') == 'prepared':
                    record.update(status='failed', error=str(exc))
        log_results(records, owner)
    for hook in queue.hooks:
        try:
            hook()
        except Exception as exc:
            delivery_log(f'attachment handoff failed: {exc}', failed=True, owner=owner)
    records = owner._artifact_results
    counts = {status: sum(r.get('status') == status for r in records)
              for status in ('prepared', 'published', 'skipped', 'failed')}
    counts['unregistered'] = int(not records and not owner._artifacts)
    owner._artifact_delivery_summary = counts
    if not any(_visible_result(r) for r in records):
        return
    delivery_log(notice(records).strip(), failed=bool(counts['failed'] or counts['skipped']), owner=owner)


def defer_mail(callback):
    queue = task_delivery()
    if queue is None or queue.mailing:
        return None
    state = {'status': 'queued', 'reason': 'waiting for task artifact delivery'}
    queue.mail.append((callback, state))
    return state


def run_deferred_mail(owner):
    queue = task_delivery(owner)
    assert queue is not None
    queue.mailing = True
    for callback, state in queue.mail:
        try:
            result = callback()
            state.clear()
            state.update(result)
        except Exception as exc:
            state.update(status='failed', error=str(exc))
            delivery_log(f'deferred mail failed: {exc}', failed=True, owner=owner)


def register(spec, task, run_id, status='success'):
    model = MODEL_ARTIFACTS.get()
    if model is not None:
        model.append(spec)
        return []
    owner = CURRENT_TASK.get()
    if owner is not None:
        owner.register_artifact(spec)
        return []
    return publish_safe(task, run_id, [spec], status)


def analysis_specs(folder, category):
    folder = Path(folder)
    members = []
    for p in sorted(folder.rglob('*')):
        rel = p.relative_to(folder)
        if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in ANALYSIS_SUFFIXES:
            # Stock-level data and delivery metadata are not analysis summaries.
            if any(x in p.name.lower() for x in ('prediction', 'package_manifest', 'parts.json', 'mail_status')):
                continue
            members.append(rel.as_posix())
    specs = [ArtifactSpec(folder, category=category, members=tuple(members), experiment=folder.name)] if members else []
    for tb in folder.rglob('tensorboard'):
        if tb.is_dir() and not any(x in EXCLUDED_DIRS - {'tensorboard'} for x in tb.relative_to(folder).parts):
            if any(tb.rglob('*tfevents*')):
                specs.append(ArtifactSpec(tb, kind='tensorboard', category=category, name=tb.relative_to(folder).as_posix(), experiment=folder.name))
    return specs


def publish_analysis(folder, category, status='success'):
    if sandboxed():
        return []
    try:
        config = Config.load()
        if not config.enabled or config.role != 'publisher':
            return []
        folder = Path(folder)
        from .core import digest
        import hashlib
        specs = analysis_specs(folder, category)
        fingerprint = hashlib.sha256()
        for spec in specs:
            paths = [spec.source / n for n in spec.members] if spec.members else sorted(p for p in spec.source.rglob('*') if p.is_file())
            for path in paths:
                fingerprint.update(str(path.relative_to(folder)).encode())
                fingerprint.update(digest(path).encode())
        records = publish_safe(category, folder.name + '__' + status + '__' + fingerprint.hexdigest()[:16], specs, status)
        return records
    except Exception as exc:
        delivery_log(f'analysis preparation failed: {exc}', failed=True)
        return [{'status': 'failed', 'error': str(exc)}]


def delivered(records):
    return bool(records) and all(r.get('status') == 'published' for r in records)


def notice(records):
    records = [r for r in records if _visible_result(r)]
    if not records:
        return ''
    counts = {status: sum(r.get('status') == status for r in records)
              for status in ('prepared', 'published', 'skipped', 'failed')}
    reasons = list(dict.fromkeys(str(r.get('error') or r.get('reason'))
                   for r in records if r.get('error') or r.get('reason')))
    text = f'\nArtifact delivery summary: total={len(records)}; ' + json.dumps(counts)
    if reasons:
        text += '; reasons=' + '; '.join(reason[:200] for reason in reasons[:3])
        if len(reasons) > 3:
            text += f'; {len(reasons) - 3} additional reasons in delivery records'
    return text + '; published means local shared-folder handoff, not confirmed remote archival.'


def handoff_bundle(metadata, folder, category, status='success', *,
                   attachment_paths: Iterable[str | Path] | Callable[[], Iterable[str | Path]] = (),
                   replacement_attachments: Callable[[], Iterable[str | Path]] | None = None):
    try:
        records = publish_analysis(folder, category, status)
        def finish():
            if not delivered(records):
                return
            from .core import atomic_json
            logs = [str(p) for p in Path(folder).glob('*.log') if p.is_file()]
            atomic_json(Path(metadata).with_suffix('.artifacts.json'), {'records': records, 'logs': logs})
            if CURRENT_TASK.get() is not None:
                from src.proj import Proj
                described = attachment_paths() if callable(attachment_paths) else attachment_paths
                old = {Path(p).resolve() for p in described}
                for files in (Proj.email_attachments, Proj.exit_files):
                    original = files.pop_all()
                    matched = any(Path(p).resolve() in old for p in original)
                    for p in original:
                        if Path(p).resolve() not in old:
                            files.append(p.path, filename=p.filename)
                    if matched:
                        for p in (replacement_attachments() if replacement_attachments else []):
                            files.append(p)
        queue = task_delivery()
        if queue is not None and not queue.flushing:
            queue.bundles[str(Path(metadata).resolve())] = records
            queue.hooks.append(finish)
        else:
            finish()
        return records
    except Exception as exc:
        delivery_log(f'bundle handoff failed: {exc}', failed=True)
        return [{'status': 'failed', 'error': str(exc)}]


def bundle_delivery(metadata):
    if sandboxed():
        return None
    queue = task_delivery()
    if queue is not None:
        pending = queue.bundles.get(str(Path(metadata).resolve()))
        if pending is not None and not delivered(pending):
            return None
    sidecar = Path(metadata).with_suffix('.artifacts.json')
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return None


def register_report(source, run_id):
    try:
        spec = ArtifactSpec(source, category='model_training')
        if CURRENT_TASK.get() is None:
            config = Config.load()
            if sandboxed() or not config.enabled or config.role != 'publisher':
                return
            from src.proj import Save
            Save.async_wait_all(caller_name='artifact_exchange')
        register(spec, 'model_training', run_id)
    except Exception as exc:
        delivery_log(f'report registration failed: {exc}', failed=True)


def model_result_specs(base, name):
    from .model_metadata import DIRECTORIES, metadata_files

    specs = []
    results = base / 'results'
    members = tuple(path.relative_to(results).as_posix() for path in sorted(results.rglob('*'))
                    if path.is_file() and not path.is_symlink()
                    and not any(part.startswith('.') for part in path.relative_to(results).parts)
                    and path.relative_to(results).parts[0] not in ('optuna', 'tensorboard', 'factor_residual')
                    and path.name not in {'fitting_output.html', 'testing_output.html', 'fitting_testing_output.html'})
    if members:
        specs.append(ArtifactSpec(results, category='model_training', members=members))
    events = results / 'tensorboard'
    if any(events.rglob('*tfevents*')):
        specs.append(ArtifactSpec(events, kind='tensorboard', category='model_training', name=name))
    for path in sorted((results / 'optuna').glob('*.sqlite3')):
        if path.is_file() and not path.is_symlink():
            specs.append(ArtifactSpec(path, kind='optuna', category='model_training'))
    files = [relative for _, relative in metadata_files(base)]
    for directory in DIRECTORIES:
        members = tuple(name.split('/', 1)[1] for name in files if name.startswith(directory + '/'))
        if members:
            specs.append(ArtifactSpec(base / directory, category='model_training', members=members, destination=directory))
    html = base / 'run_logs/html'
    captures = tuple(path.name for path in sorted(html.glob('*.html'))
                     if path.is_file() and not path.is_symlink())
    if captures:
        specs.append(ArtifactSpec(html, category='model_training', members=captures, destination='html'))
    return specs


@contextmanager
def model_run(trainer, log=None):
    """Freeze one invocation's closed writers while the model lock is held."""
    specs = []
    status = 'success'
    run_id = log.run_id if log is not None else uuid.uuid4().hex
    token = MODEL_ARTIFACTS.set(specs)
    try:
        yield
    except BaseException:
        status = 'failed'
        raise
    finally:
        MODEL_ARTIFACTS.reset(token)
        try:
            config = None if sandboxed() else Config.load()
            if config is not None and config.enabled and config.role == 'publisher':
                for callback in trainer.callback.callbacks:
                    close = getattr(callback, 'reset_writer', None)
                    if close is not None:
                        close()
                from src.proj import Save
                from .model_identity import build_identity

                Save.async_wait_all(caller_name='artifact_exchange')
                base = trainer.config.base_path
                build = build_identity(base()) if log is not None else None
                if log is not None:
                    specs.extend(model_result_specs(base(), base.full_name))
                else:
                    results = base.rslt()
                    members = tuple(p.relative_to(results).as_posix() for p in results.rglob('*') if p.is_file())
                    if members:
                        specs.append(ArtifactSpec(results, category='model_training', members=members))
                    events = base.rslt('tensorboard')
                    if any(events.rglob('*tfevents*')):
                        specs.append(ArtifactSpec(events, kind='tensorboard', category='model_training', name=base.full_name))
                unique = {}
                for spec in specs:
                    key = (spec.source, spec.kind) if spec.kind == 'optuna' else (spec.source, spec.kind, spec.destination)
                    unique.setdefault(key, replace(spec, category='model_training', experiment=base.full_name, run_id=run_id))
                if unique:
                    publish_safe('model_training', run_id, list(unique.values()), status, build=build)
            else:
                reason = 'autofix' if config is None else f'publisher inactive: enabled={config.enabled}, role={config.role}'
                record = {'status': 'skipped', 'reason': reason}
                owner = CURRENT_TASK.get()
                if owner is not None:
                    owner._artifact_results.append(record)
                log_results([record], owner)
        except Exception as exc:
            record = {'status': 'failed', 'error': f'model preparation failed: {exc}'}
            owner = CURRENT_TASK.get()
            if owner is not None:
                owner._artifact_results.append(record)
            log_results([record], owner)


def delivered_report_paths(records):
    """Remove a mail attachment only when its current bytes were actually published."""
    from .core import digest, real_tree
    paths = set()
    for record in records:
        if record.get('status') != 'published':
            continue
        for item in record.get('source_files', []):
            try:
                path = real_tree(Path(item['source']))
                if path.suffix.lower() not in ('.log', '.html') and digest(path) == item['sha256']:
                    paths.add(str(path.resolve()))
            except (ValueError, OSError):
                continue
    return paths
