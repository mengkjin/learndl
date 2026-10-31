"""Checkout locations only; no project initialization or runtime-state migration."""
from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'src').is_dir():
            return parent
    raise RuntimeError('Cannot locate the learndl checkout')


ROOT = project_root()


def config_directory(directory: Path | None = None, *, root: Path = ROOT) -> Path:
    """Map only the old built-in directory; custom config directories are unchanged."""
    root = root.resolve()
    if directory is None or directory.resolve() == root / 'runs/scheduling':
        return root / 'configs/automation'
    return directory.resolve()


def autofix_manifest(root: Path = ROOT) -> Path:
    return root / 'configs/automation/autofix.yaml'
