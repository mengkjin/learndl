#!/usr/bin/env python3
# description: Backfill missing pre-2017 daily factors and refresh repaired downstream data
# email: true
# mode: shell
# parameters:
#   start:
#     desc: First repair trading date (YYYYMMDD)
#     type: int
#     default: 20110101
#   end:
#     desc: Backfill end date; defaults to 20161230. Downstream refresh extends to latest confirmed date
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
from src.data.preprocess.factor_sequence.repair import run_repair, arguments

__all__ = ["main"]


@ScriptTool(
    "backfill_factor_sequence_pre2017",
    lock_name="update_factors",
    markdown_catcher=True,
)
def main(start=20110101, end=None, run_id=None, dry_run=False, batch_size=32):
    return run_repair("backfill", start, end, run_id, dry_run, batch_size)


if __name__ == "__main__":
    main(**arguments())
