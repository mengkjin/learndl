"""Discover local and received model builds and preserve their catalog identities."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from src.automation.artifact_exchange.core import Config, received_batches
from src.automation.artifact_exchange.model_builds import archived_builds, load
from src.shared.file_ops import real_tree
from src.shared.model_provenance import PROPERTY_FILE, model_id, read_properties

__all__ = ['ModelManagementContext', 'inventory', 'model_views']


@dataclass(frozen=True)
class ModelManagementContext:
    config: Config
    state: Path
    model_root: Path
    model_archive: Path
    instance_lock: Path | None = None
    running_instances: Callable[[], Iterable[Mapping]] | None = None


def model_views(state: Path) -> dict:
    path = state / 'model_views.json'
    return json.loads(real_tree(path).read_text()) if path.exists() else {}


def inventory(context):
    from src.proj import MACHINE
    from src.res.model.util.core.model_path import source_models

    views = model_views(context.state)
    rows = {}

    def add(host, origin, name, build, path, archived, external, preview_only=False):
        identity = model_id(dict(host=host, model=origin))
        key = sha256(json.dumps([identity, build, str(path)]).encode()).hexdigest()
        rows[key] = dict(id=key, model_id=identity, host=host, source_name=origin, name=name,
                         build=build, path=str(path), archived=archived, external=external,
                         preview_only=preview_only, hidden=views.get(identity, {}).get(build, 'active') != 'active')

    for path, index in archived_builds(context.config):
        relative = path.relative_to(context.config.results / 'external/models')
        add(index['host'], index['model'], relative.parts[-2], path.name, path,
            len(relative.parts) == 4, True)
    for path, manifest in received_batches(context.config, 'model_training'):
        origin = manifest.get('experiment') or manifest['task']
        add(manifest.get('host') or 'unknown', origin, origin, '', path.parent, False, True, True)
    for model, path, stamp in source_models(context.model_root):
        build = stamp
        with suppress(OSError, ValueError, KeyError, TypeError):
            build = read_properties(path / PROPERTY_FILE).get('build', {}).get('name') or stamp
        add(context.config.source_host, model.full_name, model.full_name, build, path, bool(stamp),
            False, (path / '.build-preview-only').exists())
    from yaml import safe_load
    from src.res.model.util.core.basic import parse_model_input
    from src.res.model.util.core.model_path import ModelPath
    from src.res.model.util.lifecycle import ModelLifecycle
    registered = ModelLifecycle.production_names()
    for row in rows.values():
        model = ModelPath(row['name'])
        row['short_name'] = model.model_name
        fields = parse_model_input(row['name'])
        row.update(module_type=fields['module_type'], module=fields['module_name'],
                   short='yes' if fields['st'] else 'no', stage='unknown', started='')
        row['host_display'] = row['host'] if row['external'] else MACHINE.name
        base = real_tree(row['path'])
        row['fingerprint'] = (load(base / 'build.json', {}).get('build', {}).get('fingerprint')
                              if row['external'] else read_properties(base / PROPERTY_FILE).get('build', {}).get('fingerprint'))
        with suppress(OSError, ValueError, KeyError, TypeError, AttributeError):
            row['started'] = read_properties(base / PROPERTY_FILE).get('initial_training', {}).get('started_at', '')
        with suppress(OSError, ValueError, KeyError, TypeError, AttributeError):
            row['stage'] = safe_load((base / 'configs/lifecycle.yaml').read_text()).get('stage', 'unknown')
        row['stage'] = ModelLifecycle.catalog_stage(row['name'], row['stage'], registered, archived=row['archived'])
    return rows
