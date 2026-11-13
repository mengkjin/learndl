"""Explicit research result adapters. Failed delivery never masks research outcomes."""
from __future__ import annotations
import contextvars
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import uuid
from typing import Any
import json
from pathlib import Path
from .core import ArtifactSpec, Config, publish_artifacts, sandboxed

MODEL_ARTIFACTS: contextvars.ContextVar[list[ArtifactSpec] | None] = contextvars.ContextVar('artifact_model', default=None)
CURRENT_TASK: contextvars.ContextVar[Any] = contextvars.ContextVar('artifact_task', default=None)
ANALYSIS_SUFFIXES = {'.pdf', '.xlsx', '.json', '.csv', '.html', '.md', '.txt', '.log'}
EXCLUDED_DIRS = {'models', 'model', 'checkpoints', 'checkpoint', 'snapshots', 'data', 'dataset', 'prepared', 'mail', 'training_mail', 'bundles', 'tensorboard', 'recovery'}


def publish_safe(task, run_id, artifacts, status='success'):
    try:
        return publish_artifacts(task, run_id, artifacts, status)
    except Exception as exc:
        print(f'[artifacts] delivery failed: {exc}', flush=True)
        return [{'status': 'failed', 'error': str(exc)}]


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
        owner = CURRENT_TASK.get()
        if owner is not None:
            owner._artifact_results.extend(records)
        return records
    except Exception as exc:
        print(f'[artifacts] analysis delivery failed: {exc}', flush=True)
        return [{'status': 'failed', 'error': str(exc)}]


def delivered(records):
    return bool(records) and all(r.get('status') == 'published' for r in records)


def notice(records):
    return '\nArtifact delivery: ' + json.dumps([{k: r[k] for k in ('id', 'status', 'error') if k in r} for r in records]) if records else ''


def handoff_bundle(metadata, folder, category, status='success'):
    try:
        records = publish_analysis(folder, category, status)
        if delivered(records):
            from .core import atomic_json
            logs = [str(p) for p in Path(folder).glob('*.log') if p.is_file()]
            atomic_json(Path(metadata).with_suffix('.artifacts.json'), {'records': records, 'logs': logs})
        return records
    except Exception as exc:
        print(f'[artifacts] bundle handoff failed: {exc}', flush=True)
        return [{'status': 'failed', 'error': str(exc)}]


def bundle_delivery(metadata):
    if sandboxed():
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
        print(f'[artifacts] report registration failed: {exc}', flush=True)


@contextmanager
def model_run(trainer):
    """Collect one model invocation, including standalone and resumed trainers."""
    specs = []
    token = MODEL_ARTIFACTS.set(specs)
    status = 'success'
    run_id = datetime.now(timezone.utc).isoformat() + '__' + uuid.uuid4().hex
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
                # On failure, close still-open writers before considering diagnostics.
                for callback in trainer.callback.callbacks:
                    close = getattr(callback, 'reset_writer', None)
                    if close is not None:
                        close()
                from src.proj import Save
                Save.async_wait_all(caller_name='artifact_exchange')
                base = trainer.config.base_path
                results = base.rslt()
                if results.exists() and any(p.is_file() for p in results.rglob('*')):
                    specs.append(ArtifactSpec(results, category='model_training'))
                events = base.snapshot('tensorboard')
                if events.exists() and any(events.rglob('*tfevents*')):
                    specs.append(ArtifactSpec(events, kind='tensorboard', category='model_training', name=base.full_name))
                specs = [replace(s, category='model_training', experiment=base.full_name, run_id=run_id) for s in specs]
                # Studies may be registered once per date; one final SQLite snapshot suffices.
                specs = list({(s.source, s.kind, s.destination): s for s in specs}.values())
                owner = CURRENT_TASK.get()
                if specs:
                    records = publish_safe('model_training', run_id, specs, status)
                    if owner is not None:
                        owner._artifact_results.extend(records)
        except Exception as exc:
            print(f'[artifacts] model delivery failed: {exc}', flush=True)


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
