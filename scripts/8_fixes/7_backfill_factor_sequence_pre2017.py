#!/usr/bin/env python3
# description: Recover a completed factor rebuild, backfill daily history and construct data blocks
# email: true
# mode: shell
# parameters:
#   source_run:
#     desc: Original rebuild run with only the data stage failed; preserved unchanged
#     type: str
#     default: rebuild_20261116T064711Z_a33e87ad
#   start:
#     desc: First repair trading date (YYYYMMDD)
#     type: int
#     default: 20110101
#   end:
#     desc: Backfill end date; defaults to the original rebuild cutoff, including all pre-2017 gaps
#     type: int
#     default: null
#   run_id:
#     desc: Reuse the same run ID to resume; leave blank to create a new run
#     type: str
#     default: null
#   dry_run:
#     desc: Write the execution plan without recalculating factors or data blocks
#     type: bool
#     default: false
#   batch_size:
#     desc: Number of trading dates per data construction batch
#     type: int
#     default: 32
"""Manual repair; no training, worklist edits or historical audit workflow."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.proj.util.script import ScriptTool
from src.data.maintenance.factor_sequence.repair import arguments
from src.data.maintenance.factor_sequence.recovery import run_recovery

__all__ = ["main"]


@ScriptTool(
    "backfill_factor_sequence_pre2017",
    lock_name="update_factors",
    markdown_catcher=True,
)
def main(
    start=20110101,
    end=None,
    run_id=None,
    dry_run=False,
    batch_size=32,
    source_run="rebuild_20261116T064711Z_a33e87ad",
):
    return run_recovery(start, end, run_id, dry_run, batch_size, source_run)


if __name__ == "__main__":
    main(**arguments(recovery=True))
