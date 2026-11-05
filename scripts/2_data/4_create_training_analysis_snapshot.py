#!/usr/bin/env python3
# coding: utf-8
# author: auto
# date: 2026-11-04
# description: Create Training Analysis Snapshot
# content: |
#   创建或更新模型训练分析快照，保存到 PATH.interim / training_analysis_snapshot。
#   resume=True 在已有快照基础上更新，复用未变化的年度数据。
#   resume=False 按指定日期范围重新计算，通过校验后原子发布。
#   每次成功更新清理旧版本，保留当前版、上一版及正在读取的版本。
# email: False
# mode: shell
# parameters:
#   resume:
#       type: [True, False]
#       desc: 是否在已有快照基础上增量更新
#       required: False
#       default: True
#   start:
#       type: int
#       desc: 开始日期 YYYYMMDD（包含当天）
#       required: False
#       default: 20170103
#   end:
#       type: int
#       desc: 结束日期 YYYYMMDD（包含当天），默认截至最新可用行情日期
#       required: False
#       default: 99991231
"""Create or update model-training analysis inputs."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from src.proj import Logger
from src.proj.util.script import ScriptTool


@ScriptTool("create_training_analysis_snapshot")
def main(resume: bool = True, start: int = 20170103, end: int = 99991231):
    from src.proj import CALENDAR, DB
    from src.res.model.util.data.training_analysis_snapshot import build_snapshot

    end = min(end, int(CALENDAR.updated()))
    dates = [
        int(date) for date in DB.dates("trade_ts", "day") if start <= int(date) <= end
    ]
    if not dates:
        raise ValueError("No available daily data in the requested interval")
    report = build_snapshot(dates, resume=resume)
    Logger.info(json.dumps(report, indent=2))
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
    parser.add_argument("--task_id", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--email", type=parse_bool, help=argparse.SUPPRESS)
    main(
        **{
            key: value
            for key, value in vars(parser.parse_args()).items()
            if value is not None
        }
    )
