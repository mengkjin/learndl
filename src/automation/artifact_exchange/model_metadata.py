"""Publication of model configuration and diagnostic files alongside results."""
from __future__ import annotations

from contextlib import contextmanager
import json
from tempfile import TemporaryDirectory
from pathlib import Path

from .core import ArtifactSpec, atomic_json, component, digest, real_tree

DIRECTORIES = {
    'configs': {'.yaml', '.yml', '.json', '.toml'},
    'logs': {'.log', '.txt', '.json', '.jsonl', '.md', '.html', '.yaml', '.yml'},
}
INVENTORY = '.model-metadata/inventory.json'
BACKFILL_TASK = 'model_metadata_backfill'


def source_models(root):
    """Yield complete identities and actual paths, including archived copies."""
    from src.res.model.util import ModelPath

    for base in sorted(root.glob('*/*')):
        try:
            real_tree(base)
            if not base.is_dir() or base.name.startswith('.'):
                continue
            if base.parent.name == 'archive':
                name, _, stamp = base.name.rpartition('.')
                if len(stamp) == 14 and stamp.isdigit() and (model := ModelPath(name)):
                    yield model, base, stamp
            elif model := ModelPath(base):
                yield model, base, ''
        except (AssertionError, OSError, ValueError):
            continue


def metadata_files(base):
    for directory, suffixes in DIRECTORIES.items():
        folder = base / directory
        if not folder.exists():
            continue
        for path in sorted(real_tree(folder).rglob('*')):
            relative = path.relative_to(base)
            if (not any(part.startswith('.') for part in relative.parts)
                    and path.suffix.lower() in suffixes and not path.is_symlink() and path.is_file()):
                yield real_tree(path), relative.as_posix()


def read_inventory(path):
    try:
        return json.loads(real_tree(path).read_text())
    except FileNotFoundError:
        return {}


@contextmanager
def metadata_specs(config, base, model, *, full=False):
    """Describe a complete inventory, with bytes for members changed since the prior snapshot."""
    current = {relative: digest(path) for path, relative in metadata_files(base)}
    baseline = real_tree(config.state / 'model_metadata' / component(config.source_host) / (component(model) + '.json'))
    previous = {} if full else read_inventory(baseline)
    specs = []
    if current == previous:
        yield specs, baseline, current
        return
    for directory in DIRECTORIES:
        members = tuple(name.split('/', 1)[1] for name, sha in current.items()
                        if name.startswith(directory + '/') and previous.get(name) != sha)
        if members:
            specs.append(ArtifactSpec(base / directory, category='model_training', members=members, destination=directory))
    with TemporaryDirectory(prefix='learndl-model-metadata-') as temp:
        index = Path(temp).resolve() / 'inventory.json'
        atomic_json(index, current)
        specs.append(ArtifactSpec(index, category='model_training', destination='.model-metadata'))
        yield specs, baseline, current
