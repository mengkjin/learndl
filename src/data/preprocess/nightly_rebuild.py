"""Checkpointed DataBlock rebuilds using the normal production preprocessors."""
import os
import shutil

import numpy as np

from src.data.preprocess.processors import PrePros
from src.data.util import DataBlock
from src.proj import Dates, Logger, Save
from src.proj.db.io.mmap import ArrayMemoryMap, ArrayMeta
from src.proj.db.io.torch import torch_save


def rebuild_block(job, revision: str, frame: str, key: str):
    processor = PrePros.get_processor(key, frame, vb_level=1)
    canonical = DataBlock.path_preprocess(key, frame, find_if_not_exists=False)
    interrupted = canonical.with_name(f'{canonical.name}.old')
    if interrupted.exists() and not canonical.exists():
        interrupted.rename(canonical)
    base = canonical.parent / f'.{revision}.{key}'
    stage = f'{revision}_block'
    item = f'{frame}/{key}'
    if job.read(stage, item) and canonical.is_dir():
        return
    bounds = job.read(stage, f'{item}/bounds')
    if bounds is None:
        bounds = [processor.load_start, processor.load_end]
        job.mark(stage, f'{item}/bounds', bounds)
    elif (processor.load_end > bounds[1] and not (base / 'publish.mmap').exists()
          and not job.read(stage, f'{item}/publish_ready')):
        bounds[1] = processor.load_end
        job.mark(stage, f'{item}/bounds', bounds)
    dates = list(map(int, Dates(*bounds)))
    groups = {}
    for day in dates:
        groups.setdefault(day // 100, []).append(day)

    shards = []
    for month, days in groups.items():
        job.guard()
        path = base / f'{month}-{days[-1]}.mmap'
        unit = f'{item}/{path.stem}'
        record = job.read(stage, unit)
        if record is None or (record and not path.is_dir()):
            block = processor.pre_process(days[0], days[-1])
            record = not block.empty
            if record:
                Save.mmap(block.values, path, index={'date': block.date, 'secid': block.secid,
                                                    'feature': block.feature})
            job.mark(stage, unit, record)
            Logger.stdout(f'Nightly DataBlock {item}: prepared {month}')
        if record:
            shards.append(path)
    if not shards:
        raise RuntimeError(f'Nightly DataBlock {item}: no output')

    indexes = [ArrayMemoryMap.load(path, values=False, index=True)['index'] for path in shards]
    coords = {
        'secid': np.unique(np.concatenate([index['secid'] for index in indexes])),
        'date': np.concatenate([index['date'] for index in indexes]),
        'feature': indexes[0]['feature'],
    }
    if len(coords['date']) == 0 or not np.all(np.diff(coords['date']) > 0):
        raise ValueError(f'Nightly DataBlock {item}: missing or overlapping date shards')
    meta = ArrayMeta.from_json(ArrayMemoryMap(shards[0]).meta_path)
    shape = (len(coords['secid']), len(coords['date']), *meta.shape[2:])
    bundle = ArrayMemoryMap(base / 'publish.mmap')
    finished_bundle = ArrayMemoryMap(canonical)
    published = job.read(stage, f'{item}/publish_ready') and not bundle.path.exists() and canonical.is_dir()
    if not published:
        bundle.prepare()
        if not bundle.data_path.exists() or not bundle.meta_path.exists():
            with job.db:
                job.db.execute('DELETE FROM checkpoints WHERE stage=? AND item LIKE ?',
                               (stage, f'{item}/copy/%'))
            with bundle.data_path.open('wb') as data:
                data.truncate(int(np.prod(shape)) * np.dtype(meta.dtype).itemsize)
            torch_save(coords, bundle.path / 'index.pt')
            ArrayMeta('Tensor', meta.dtype, shape).to_json(bundle.meta_path)
        values = np.memmap(bundle.data_path, mode='r+', dtype=meta.dtype, shape=shape)
        for path, index in zip(shards, indexes):
            job.guard()
            if job.read(stage, f'{item}/copy/{path.stem}'):
                continue
            start = np.searchsorted(coords['date'], index['date'][0])
            end = start + len(index['date'])
            if not np.array_equal(index['feature'], coords['feature']):
                raise ValueError(f'Nightly DataBlock {item}: inconsistent features at {path.stem}')
            rows = np.searchsorted(coords['secid'], index['secid'])
            values[:, start:end] = np.nan
            values[rows, start:end] = ArrayMemoryMap(path).view()
            values.flush()
            with bundle.data_path.open('rb') as handle:
                os.fsync(handle.fileno())
            job.mark(stage, f'{item}/copy/{path.stem}')
        del values
        job.guard()
        original = DataBlock.path_preprocess(key, frame)
        backup = canonical.parent / f'.{revision}_backup' / original.name
        if original.exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            if original.is_dir():
                staged = backup.with_name(f'{backup.name}.partial')
                if staged.exists():
                    shutil.rmtree(staged)
                shutil.copytree(original, staged, copy_function=os.link)
                staged.rename(backup)
            else:
                os.link(original, backup)
        job.mark(stage, f'{item}/publish_ready')
        ArrayMemoryMap._publish(canonical, bundle.path)
    index = ArrayMemoryMap.load(canonical, values=False, index=True)['index']
    if not all(np.array_equal(coords[axis], index[axis]) for axis in coords):
        raise ValueError(f'Nightly DataBlock {item}: published coordinates differ from shards')
    if frame == 'fit' and key in ('15m', '30m', '30mcont', '15mt', '30mt'):
        job.guard()
        values = finished_bundle.view()
        historical = DataBlock(values, **index).slice_date(processor.hist_start, processor.hist_end)
        processor.save_norm(historical)
    job.mark(stage, item, {'path': str(canonical), 'dates': len(coords['date'])})
    shutil.rmtree(base)
    Logger.stdout(f'Nightly DataBlock {item}: published {len(coords["date"])} dates')


def retire_60m(job, revision):
    from src.data.preprocess.hist_norm import PreProHistNorm
    from src.proj import PATH
    candidates = [PreProHistNorm.norm_path('60m')]
    for frame in ('fit', 'predict'):
        for alias in DataBlock.data_type_alias('60m'):
            candidates.extend(PATH.datablock / frame / f'X_{alias}{suffix}'
                              for suffix in ('.mmap', '.pt', '.feather', '.mmap.old', '.mmap.staging'))
    candidates.extend(PATH.histnorm / 'fit' / f'X_{alias}.pt' for alias in DataBlock.data_type_alias('60m'))
    for path in dict.fromkeys(candidates):
        job.guard()
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            continue
        job.mark(f'{revision}_retired', str(path))
        Logger.stdout(f'Nightly retired DataBlock output {path}')
