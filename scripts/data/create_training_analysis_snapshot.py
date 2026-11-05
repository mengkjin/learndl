#!/usr/bin/env python3
"""Create or update model-training analysis inputs."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(resume: bool = True, start: int = 20170103, end: int = 99991231):
    from src.proj import CALENDAR, DB
    from src.res.factor.training_analysis_snapshot import build_snapshot

    dates = [
        int(date)
        for date in DB.dates("trade_ts", "day")
        if start <= int(date) <= min(end, int(CALENDAR.updated()))
    ]
    if not dates:
        raise ValueError("No available daily data in the requested interval")
    report = build_snapshot(dates, resume=resume)
    print(json.dumps(report, indent=2))
    return report


def parse_bool(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("resume must be true or false")
    return value.lower() == "true"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        type=parse_bool,
        default=True,
        help="true updates existing coverage; false rebuilds the requested interval",
    )
    parser.add_argument("--start", type=int, default=20170103)
    parser.add_argument("--end", type=int, default=99991231)
    main(**vars(parser.parse_args()))
