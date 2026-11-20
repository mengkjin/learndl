"""Consolidate old timestamp batches by source and model name."""
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import portalocker

from src.automation.artifact_exchange.core import (
    Config, archived_batches, atomic_json, digest, experiment_members, fsync_dir,
    locked, publish_artifacts, real_tree, received_member, send_prepared,
)
from src.automation.artifact_exchange.model_builds import (
    ALIAS, INDEX, apply_update, legacy_target, load, merge_legacy_log,
)
from src.automation.artifact_exchange.research import model_result_specs
from src.automation.version_patch import load_patch_support
from src.shared.model_provenance import LOG_PHASES
from src.res.model.util.core.model_path import source_models
from src.res.model.util.model_identity import build_identity


def move(source, target):
    source, target = real_tree(source), real_tree(target)
    if not source.exists():
        if target.exists():
            return
        raise FileNotFoundError(source)
    if target.exists():
        raise ValueError(f'Migration backup already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    fsync_dir(source.parent)
    fsync_dir(target.parent)


def merge_source_logs(base, backup):
    for phase in LOG_PHASES:
        original = real_tree(base / 'run_logs' / f'{phase}.md')
        if original.is_file():
            merge_legacy_log(real_tree(base / 'logs' / original.name), original)
            move(original, backup / original.name)


def migrate_receiver(config, root, backup):
    mapping = {}
    batches = sorted(archived_batches(config, 'model_training'), key=lambda pair: pair[1].get('created_at', ''))
    for folder, manifest in batches:
        target = legacy_target(folder.parent, manifest)
        members = experiment_members(manifest)
        canonical = {item['path']: received_member(manifest, item['path'], members, folder=folder)
                     for item in manifest['files']}
        apply_update(target, folder, manifest, canonical, members, legacy=True)
        mapping[folder] = target

    for path in (config.results / 'external/models').glob('*/*/run_logs'):
        target_name = load(path.parent / ALIAS)
        if not target_name:
            info = load(path / 'model.json')
            if not info or not any((path / f'{phase}.md').is_file() for phase in LOG_PHASES):
                continue
            target = legacy_target(path.parent, {})
            target.mkdir(exist_ok=True)
            index = {'host': info['host'], 'model': info['model'],
                     'build': {'name': target.name, 'estimated': True, 'legacy_attribution': 'source_and_model_name'},
                     'files': {}, 'runs': {}, 'created_at': '19700101T000000000000Z', 'status': 'success'}
        else:
            target = path.parent / target_name
            index = load(target / INDEX)
            if index is None:
                raise ValueError(f'Missing build index at {target}')
        for phase in LOG_PHASES:
            original = real_tree(path / f'{phase}.md')
            relative = f'logs/{phase}.md'
            if not original.is_file() or index['files'].get(relative, {}).get('sequence', 0):
                continue
            destination = real_tree(target / relative)
            merge_legacy_log(destination, original)
            if destination.exists():
                index['files'][relative] = {
                    'path': relative, 'sha256': digest(destination), 'size': destination.stat().st_size,
                    'kind': 'reports', 'stamp': index['created_at'], 'sequence': 0, 'package': '',
                }
        atomic_json(target / INDEX, index)
        mapping[path] = target / 'logs'

    layout = load_patch_support('0.7.4', 'results')
    layout.update_state(config, root / '.local_resources/learndl_monitor', mapping)
    for source in mapping:
        if source.name != 'run_logs':
            move(source, backup / 'received' / source.relative_to(config.results))
    for source in mapping:
        if source.name == 'run_logs':
            move(source, backup / 'received' / source.relative_to(config.results))
    return len(batches)


def backfill(config, models):
    if not config.enabled or config.role != 'publisher':
        return {}
    prior = set()
    for path in config.state.glob('outbox/*/delivery.json'):
        record = load(path, {})
        manifest = record.get('manifest', {})
        if (manifest.get('host') == config.source_host and manifest.get('category') == 'model_training'
                and 'model_build' not in manifest and record.get('status') in ('prepared', 'published')):
            prior.add(manifest.get('experiment'))
    preferred = {}
    for model, base, archived in sorted(models, key=lambda row: (not row[2], row[2])):
        preferred[model.full_name] = base
    result = {}
    for name in sorted(prior - {None}):
        if name not in preferred:
            result[name] = {'status': 'missing'}
            continue
        base = preferred[name]
        build = build_identity(base)
        specs = [replace(spec, experiment=name) for spec in model_result_specs(base, name)]
        records = publish_artifacts('model_build_backfill', '0.7.7-model-build-backfill', specs,
                                    config=config, build=build, adopt_legacy=True, prepare_only=True)
        if not records or records[0]['status'] not in ('prepared', 'published'):
            result[name] = records[0] if records else {'status': 'failed', 'error': 'No artifacts prepared'}
            continue
        record = records[0]
        outcome = send_prepared([dict(record)], config)[0]
        if outcome['status'] != 'published':
            package = config.state / 'outbox' / record['outbox_key'] / 'package.zip'
            if package.is_file() and digest(package) == record['sha256']:
                outcome['status'] = 'pending'
        result[name] = {key: outcome[key] for key in ('id', 'status', 'error') if key in outcome}
    return result


def apply(ctx):
    from src.proj import Logger, PATH
    from src.res.model.util.lifecycle import ModelLifecycle

    config = Config.load()
    models = list(source_models(PATH.model))
    for _, base, _ in models:
        with ModelLifecycle(base).lock():
            build_identity(base)
            merge_source_logs(base, ctx.backup_dir / 'source' / base.relative_to(PATH.model))

    monitor = ctx.root / '.local_resources/learndl_monitor'
    with ExitStack() as stack:
        for path in (monitor / 'model_preview', monitor / 'mail_archive'):
            if path.is_dir():
                stack.enter_context(portalocker.Lock(str(path / 'operations.lock'), timeout=0))
        stack.enter_context(locked(config, 'receive'))
        if (monitor / 'experiments').is_dir():
            stack.enter_context(portalocker.Lock(str(monitor / 'experiments/operations.lock'), timeout=0))
        from src.api.learndl_monitor.experiments import Instances

        if any((source := row.get('archive_source')) and Path(source).is_relative_to(config.results / 'external/models')
               for row in Instances(monitor).discover()):
            raise RuntimeError('Stop running synchronized model experiments before migration')
        count = migrate_receiver(config, ctx.root, ctx.backup_dir)
        if config.role == 'receiver':
            atomic_json(config.state / 'model_builds_ready.json', {'legacy_batches': count})

    result = backfill(config, models)
    report = {'source_models': len(models), 'legacy_batches': count, 'backfill': result}
    atomic_json(ctx.backup_dir / 'model-builds.json', report)
    Logger.stdout(f'Initialized {len(models)} model builds and consolidated {count} archived batches')
    if any(item['status'] == 'failed' for item in result.values()):
        raise ValueError('Some model builds could not be prepared; see model-builds.json')
    atomic_json(config.state / 'model_builds_ready.json', report)


def verify():
    config = Config.load()
    return (config.state / 'model_builds_ready.json').is_file() and not next(
        archived_batches(config, 'model_training'), None)
