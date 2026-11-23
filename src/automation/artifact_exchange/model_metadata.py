"""Publication of model configuration and diagnostic files alongside results."""
from __future__ import annotations

from contextlib import contextmanager
import json
from tempfile import TemporaryDirectory
from pathlib import Path

from .core import ArtifactSpec, atomic_json, component, digest, real_tree

__all__ = [
    'DIRECTORIES',
    'INVENTORY',
    'BACKFILL_TASK',
    'metadata_files',
    'migrate_model_html',
    'read_inventory',
    'metadata_baseline',
    'metadata_specs',
]

DIRECTORIES = {
    'configs': {'.yaml', '.yml', '.json', '.toml'},
    'logs': {'.log', '.txt', '.json', '.jsonl', '.md', '.html', '.yaml', '.yml'},
}
INVENTORY = '.model-metadata/inventory.json'
BACKFILL_TASK = 'model_metadata_backfill'


def migrate_model_html(base):
    """Move legacy source captures under logs; callers hold the model lifecycle lock."""
    old = real_tree(base / 'run_logs/html')
    if not old.is_dir():
        return
    target = real_tree(base / 'logs/html')
    for source in sorted(old.glob('*.html')):
        source = real_tree(source)
        if not source.is_file():
            continue
        target.mkdir(parents=True, exist_ok=True)
        destination = real_tree(target / source.name)
        if destination.exists() and digest(destination) != digest(source):
            destination = real_tree(target / f'{source.stem}.legacy-{digest(source)}.html')
        if destination.exists():
            if digest(destination) != digest(source):
                raise ValueError(f'Conflicting legacy HTML: {destination}')
            source.unlink()
        else:
            source.rename(destination)
    for folder in (old, old.parent):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


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


def metadata_baseline(config, model):
    return real_tree(config.state / 'model_metadata' / component(config.source_host) / (component(model) + '.json'))


@contextmanager
def metadata_specs(config, base, model, *, full=False):
    """Describe a complete inventory, with bytes for members changed since the prior snapshot."""
    current = {relative: digest(path) for path, relative in metadata_files(base)}
    baseline = metadata_baseline(config, model)
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
