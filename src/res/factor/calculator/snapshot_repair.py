"""Repair legacy field labels on stock-factor snapshots during batch normalization."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import shutil

import pandas as pd
import pyarrow as pa

from src.proj import Logger
from src.proj.db.io.snapshot import snapshot_active


def repair_level_zero(path: Path, name: str):
    if snapshot_active():
        raise ValueError(f'{path}: cannot repair level_0 inside an immutable read snapshot')
    # Concat adds nulls to clean days too. Inspect the file schema before reading rows.
    with pa.memory_map(str(path), 'r') as source:
        if 'level_0' not in pa.ipc.open_file(source).schema.names:
            return

    import fcntl
    with path.open('rb') as source:
        fcntl.flock(source, fcntl.LOCK_EX)
        if not os.path.samestat(os.fstat(source.fileno()), path.stat()):
            # Another reader repaired or a producer replaced the file. Loads rereads it.
            return
        frame = pd.read_feather(source)
        if 'level_0' not in frame:
            return
        unexpected = [c for c in frame if c not in ('level_0', 'secid', 'date', name)
                      and not c.startswith(name + '__source_')]
        if unexpected or name not in frame or 'secid' not in frame:
            raise ValueError(f'{path}: unexpected stored factor columns {list(frame.columns)}')
        if (not pd.api.types.is_numeric_dtype(frame[name]) or not pd.api.types.is_integer_dtype(frame.secid)
                or frame.secid.hasnans or not frame.secid.is_unique):
            raise ValueError(f'{path}: invalid signal dtype or secid in level_0 snapshot')
        labels = frame['level_0']
        fields = labels.dropna().unique()
        if (len(fields) != 1 or not isinstance(fields[0], str) or not fields[0].strip()
                or (labels.isna() & frame[name].notna()).any()):
            raise ValueError(f'{path}: unexpected level_0 labels or missing label for a populated signal')
        clean = frame.drop(columns='level_0')
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            clean.to_feather(temporary)
            pd.testing.assert_frame_equal(pd.read_feather(temporary), clean, check_exact=True)
            shutil.copymode(path, temporary)
            if not os.path.samestat(os.fstat(source.fileno()), path.stat()):
                raise RuntimeError(f'{path} changed during level_0 repair')
            backup = path.with_name(f'{path.name}.{temporary.name}.level_0.bak')
            os.link(path, backup)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        Logger.stdout(f'Removed level_0 from {path}, original saved at {backup}', vb_level=0)
