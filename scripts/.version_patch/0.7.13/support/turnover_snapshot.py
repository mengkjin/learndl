"""Repair the June 22 turnover snapshot left by the legacy MultiIndex writer."""
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from src.automation.console import logger
from src.res.factor.calculator import FactorCalculator


DATE = 20260622
NAME = 'turn_1m'


def _signal(frame, path):
    columns = ['secid', NAME]
    if 'level_0' in frame:
        columns.append('level_0')
    if sorted(frame.columns) != sorted(columns):
        raise ValueError(f'{path}: unexpected columns {list(frame.columns)}')
    if (frame.empty or not pd.api.types.is_integer_dtype(frame.secid)
            or frame.secid.isna().any() or not frame.secid.is_unique
            or not pd.api.types.is_numeric_dtype(frame[NAME])):
        raise ValueError(f'{path}: invalid turnover snapshot keys or signal dtype')
    if 'level_0' in frame:
        labels = frame['level_0']
        tagged = labels.isin(['turn_fr'])
        # The legacy writer reindexed to the day's universe after resetting the MultiIndex.
        padded = labels.isna() & frame[NAME].isna()
        invalid = ~(tagged | padded)
        if invalid.any():
            samples = frame.loc[invalid, ['secid', 'level_0', NAME]].head(3).to_dict('records')
            raise ValueError(f'{path}: unexpected level_0 in {invalid.sum()} rows, examples={samples!r}')
        if not tagged.any():
            raise ValueError(f'{path}: no turn_fr labels to identify the legacy snapshot')
        return frame.drop(columns='level_0')
    return frame


def _paths(ctx):
    path = FactorCalculator.get(NAME).target_path(DATE)
    return path, ctx.backup_dir / NAME / path.name


def apply(ctx):
    path, backup = _paths(ctx)
    if not path.exists():
        return
    frame = pd.read_feather(path)
    signal = _signal(frame, path)
    if 'level_0' in frame:
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(path, backup)
        pd.testing.assert_frame_equal(pd.read_feather(backup), frame, check_exact=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            signal.to_feather(temporary)
            pd.testing.assert_frame_equal(pd.read_feather(temporary), signal, check_exact=True)
            shutil.copymode(path, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info('Repaired %s (%d rows), original at %s', path, len(signal), backup)
    verify(ctx)


def verify(ctx):
    path, backup = _paths(ctx)
    if not path.exists():
        if backup.exists():
            raise FileNotFoundError(path)
        return True
    frame = pd.read_feather(path)
    _signal(frame, path)
    if 'level_0' in frame:
        raise ValueError(f'{path}: legacy level_0 is still present')
    if backup.exists():
        expected = pd.read_feather(backup).drop(columns='level_0')
        pd.testing.assert_frame_equal(frame, expected, check_exact=True)
    return True
