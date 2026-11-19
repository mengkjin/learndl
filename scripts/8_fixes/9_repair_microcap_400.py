#!/usr/bin/env python3
# description: Repair adjusted prices, listing references, microcap history and market inputs
# email: true
# mode: shell
# parameters:
#   start:
#     type: int
#     default: 20050101
#     desc: First index return date to recompute
#   end:
#     type: int
#     default: null
#     desc: Last return date; defaults to latest confirmed trading day
#   validate_only:
#     type: bool
#     default: false
#     desc: Calculate and validate staged outputs without replacing live data
"""Repair microcap history and market dumps in an idle data-maintenance window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.proj.util.script import ScriptTool
from src.data.preprocess.repair_microcap import repair_all

__all__ = ["main"]


@ScriptTool(
    "repair_microcap_400",
    lock_name="daily_update",
    lock_timeout=1,
    markdown_catcher=True,
)
def main(start=20050101, end=None, validate_only=False):
    return repair_all(start, end, validate_only)


if __name__ == "__main__":

    def boolean(value):
        if str(value).lower() not in ("true", "false"):
            raise argparse.ArgumentTypeError("Expected true or false")
        return str(value).lower() == "true"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=20050101)
    parser.add_argument("--end", type=int)
    parser.add_argument(
        "--validate-only",
        "--validate_only",
        type=boolean,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument("--task_id", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--email", type=boolean, help=argparse.SUPPRESS)
    main(**{k: v for k, v in vars(parser.parse_args()).items() if v is not None})
