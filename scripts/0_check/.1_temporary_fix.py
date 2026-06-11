#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2026-05-26
# description: Clean corrupted factor stats files
# content: |
#   Scan export/factor_stats_daily and export/factor_stats_weekly.
#   Try reading each dataframe file; if reading fails (e.g. ArrowInvalid: file too small),
#   delete the corrupted file so factor stats update can continue.
# email: True
# mode: shell
# parameters:
#   dry_run:
#       type: [True, False]
#       desc: if True, only report corrupted files without deleting
#       required: False
#       default: True
#   min_bytes:
#       type: int
#       desc: treat files smaller than this as corrupted
#       required: False
#       default: 1024

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.proj import Logger , DB , PATH
from src.proj.util.script import ScriptTool

@ScriptTool('temporary_fix')
def main(dry_run: bool = True, min_bytes: int = 1024, **kwargs):
    """
    Remove corrupted factor-stats export files.

    Notes:
      - This is a remediation for already-corrupted files. Root-cause fixes should ensure
        atomic write / locking when saving stats.
      - Deleting a factor's stats file may cause that factor to recompute historical stats.
    """

    def _iter_stats_dirs() -> Iterable[Path]:
        yield PATH.export.joinpath("factor_stats_daily")
        yield PATH.export.joinpath("factor_stats_weekly")

    def _try_read(path: Path) -> None:
        if DB.DF_SUFFIX == "feather":
            pd.read_feather(path)
        else:
            pd.read_parquet(path)

    dry_run = bool(dry_run)
    min_bytes = int(min_bytes)

    scanned = 0
    ok = 0
    bad = 0
    deleted = 0
    bad_paths: list[Path] = []

    for d in _iter_stats_dirs():
        if not d.exists():
            Logger.skipping(f"Stats export dir not found: {d}")
            continue
        paths = sorted(d.glob(f"*.{DB.DF_SUFFIX}"))
        Logger.note(f"Scanning {d} ({len(paths)} files, suffix=.{DB.DF_SUFFIX})")
        for p in paths:
            scanned += 1
            try:
                if p.stat().st_size < min_bytes:
                    raise ValueError(f"File too small ({p.stat().st_size} bytes) < {min_bytes}")
                _try_read(p)
                ok += 1
            except Exception as e:
                bad += 1
                bad_paths.append(p)
                Logger.alert1(f"Corrupted stats file: {p} ({type(e).__name__}: {e})")
                if not dry_run:
                    try:
                        p.unlink(missing_ok=True)
                        deleted += 1
                    except Exception as del_e:
                        Logger.error(f"Failed to delete {p}: {type(del_e).__name__}: {del_e}")

    Logger.note(
        f"Done. scanned={scanned}, ok={ok}, bad={bad}, "
        f"{'would_delete' if dry_run else 'deleted'}={deleted if not dry_run else bad}"
    )
    if bad_paths:
        Logger.stdout(f"Bad files (first 20): {bad_paths[:20]}", indent=1)
        
if __name__ == '__main__':
    main()
        
    