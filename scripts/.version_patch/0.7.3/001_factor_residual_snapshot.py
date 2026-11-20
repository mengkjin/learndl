# coding: utf-8
# description: Move factor residual intermediates from model results to snapshots.
# email: False
# mode: os

from src.automation.artifact_exchange.core import Config, atomic_json, category_root, fsync_dir, locked, real_tree
from src.automation.patch_diagnostics import check_open_files


def candidates(models, results):
    received = category_root(results, 'model_training')
    return sorted({
        *models.glob('*/*/results/factor_residual'),
        *models.glob('*/results/factor_residual'),
        *received.glob('*/*/results/factor_residual'),
    })


def _move(source, root):
    target = source.parent.parent / 'snapshot/factor_residual'
    real_tree(source)
    real_tree(target)
    if not source.is_dir():
        raise ValueError(f'Expected a directory: {source}')
    if target.exists():
        raise FileExistsError(f'Snapshot already exists; both directories retained: {source} and {target}')
    for entry in source.rglob('*'):
        if not real_tree(entry).is_dir() and not entry.is_file():
            raise ValueError(f'Unexpected special file: {entry}')
    check_open_files(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    fsync_dir(source.parent)
    fsync_dir(target.parent)
    return {'source': str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
            'target': str(target.relative_to(root)) if target.is_relative_to(root) else str(target)}


def apply(ctx):
    from src.proj import PATH

    config = Config.load()
    report = ctx.backup_dir / 'factor-residual.json'
    moved = []
    with locked(config, 'receive'):
        for source in candidates(PATH.model, config.results):
            try:
                moved.append(_move(source, ctx.root))
            except (OSError, ValueError, RuntimeError) as exc:
                atomic_json(report, {'moved': moved, 'blocked': str(exc)})
                raise
            atomic_json(report, {'moved': moved})
        atomic_json(report, {'moved': moved})


def verify(ctx):
    from src.proj import PATH

    return not candidates(PATH.model, Config.load().results)
