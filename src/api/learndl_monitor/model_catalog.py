"""Derive model attributes and configuration versions from the file catalog."""
from __future__ import annotations
from collections import defaultdict
from contextlib import suppress

from src.automation.artifact_exchange.model_metadata import read_inventory
from src.shared.model_provenance import PROPERTY_FILE, read_properties

__all__ = ['describe_models']


def versions(rows):
    batches = defaultdict(list)
    local = {}
    for row in rows:
        name = row.get('relative', '')
        if row['type'] == 'metadata_index' or name.startswith(('configs/', 'logs/')):
            if row.get('batch'):
                batches[row['batch']].append(row)
            else:
                local[name] = row

    if local:
        yield {'batch': '', 'files': [r['id'] for r in local.values()]}, local

    found = {}
    for batch, items in sorted(batches.items()):
        index = next((r for r in items if r['type'] == 'metadata_index'), None)
        members = [r for r in items if r is not index]
        for row in members:
            found[row['relative'], row['sha']] = row
        if index:
            try:
                inventory = read_inventory(index['path'])
                current = {name: found[name, sha] for name, sha in inventory.items() if (name, sha) in found}
                missing = len(inventory) - len(current)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
        else:
            current = {r['relative']: r for r in members}
            missing = 0
        yield {'batch': batch, 'historical': items[0].get('historical', False),
               'files': [r['id'] for r in current.values()], 'missing': missing}, current


def _describe_builds(rows, identifier):
    from src.res.model.util.lifecycle import ModelLifecycle
    registered = ModelLifecycle.production_names()
    groups = defaultdict(list)
    for row in rows:
        if row.get('model') and not row.get('trashed'):
            groups[identifier(row)].append(row)
    models = {}
    for key, files in groups.items():
        row = files[0]
        model, host = row['model'], row['host']
        short = model.startswith('st@')
        parts = model.removeprefix('st@').split('@')
        properties, stage = {}, None
        available = list(versions(files))
        if available:
            _, latest = available[-1]
            if prop := latest.get(PROPERTY_FILE):
                with suppress(OSError, ValueError, TypeError, KeyError):
                    properties = read_properties(prop['path'])
            if lifecycle := latest.get('configs/lifecycle.yaml'):
                from yaml import safe_load, YAMLError

                with suppress(OSError, ValueError, TypeError, AttributeError, YAMLError):
                    value = safe_load(lifecycle['path'].read_text()).get('stage')
                    if value in ('exploration', 'validation', 'production'):
                        stage = value
        stage = ModelLifecycle.catalog_stage(row.get('display_name', model), stage, registered,
                                             archived=row.get('physically_archived', False))
        initial = properties.get('initial_training', {})
        training = [p for p in (initial, initial.get('latest_full_training', {}),
                                properties.get('resume_training', {}))
                    if p.get('finished_at') and p.get('status') in ('success', 'estimated')]
        models[key] = {
            'id': key, 'host': host, 'name': row.get('display_name', model), 'source_name': model,
            'physically_archived': row.get('physically_archived', False), 'archive_stamp': row.get('archive_stamp'),
            'module_type': parts[0] if parts[0] in ('nn', 'boost', 'factor', 'linear') else None,
            'module': parts[1] if len(parts) > 1 else None, 'short_test': short if len(parts) > 1 else None,
            'stage': stage, 'experiments': sorted({r['type'] for r in files} & {'tensorboard', 'optuna'}),
            'properties': properties, 'last_train': max(training, key=lambda p: p['finished_at'], default={}),
            'versions': [public for public, _ in reversed(available)],
        }
    return models


def describe_models(rows, identifier):
    by_build = _describe_builds(rows, lambda row: (identifier(row), row.get('build', '')))
    grouped = defaultdict(list)
    for (model_id, build), description in by_build.items():
        grouped[model_id].append(description | {'id': model_id, 'build': build})

    def order(description):
        prefix, _, suffix = description['build'].rpartition('.')
        return (prefix, int(suffix) if suffix.isdigit() else -1, description['build'])

    result = {}
    for model_id, builds in grouped.items():
        builds.sort(key=order, reverse=True)
        result[model_id] = builds[0] | {'builds': builds}
    return result
