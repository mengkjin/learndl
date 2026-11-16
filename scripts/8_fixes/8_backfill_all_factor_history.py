#!/usr/bin/env python3
# description: Fill all historical missing factor values and daily/weekly stats serially
# email: true
# mode: shell
# parameters:
#   start:
#     desc: First date; defaults to configured factor history start
#     type: int
#     default: null
#   end:
#     desc: Last date; defaults to latest confirmed trading date
#     type: int
#     default: null
#   batch_size:
#     desc: Maximum statistics dates per serial batch
#     type: int
#     default: 32
#   dry_run:
#     desc: Print existing gaps without writing (stats for new values appear after actual repair)
#     type: bool
#     default: false
"""Missing-only repair, resumable by rerunning; preserves existing factor frequency."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.proj.util.script import ScriptTool
from src.res.factor.calculator.backfill import run_backfill


@ScriptTool('backfill_all_factor_history', lock_name='update_factors', markdown_catcher=True)
def main(start=None, end=None, batch_size=32, dry_run=False):
    return run_backfill(start, end, batch_size, dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true')
    main(**vars(parser.parse_args()))
