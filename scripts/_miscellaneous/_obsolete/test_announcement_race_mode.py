#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: codex
# date: 2026-04-30
# description: Test Announcement Race Mode
# content: 测试公告异步竞速模式，观测winner、取消与temp清理行为
# email: False
# mode: shell
# parameters:
#   start:
#       type: int
#       desc: start date
#       required: False
#       default: 20260331
#   end:
#       type: int
#       desc: end date
#       required: False
#       default: 20260401
#   workers:
#       type: int
#       desc: scheduler workers
#       required: False
#       default: 10
#   race_ratio:
#       type: float
#       desc: race trigger ratio
#       required: False
#       default: 1.0
#   min_race_tasks:
#       type: int
#       desc: minimum threshold for race mode
#       required: False
#       default: 999
#   max_replicas_per_task:
#       type: int
#       desc: max replicas per task
#       required: False
#       default: 5
#   max_total_inflight_per_exchange:
#       type: int
#       desc: max inflight per exchange
#       required: False
#       default: 20

from src.proj import Logger
from src.proj.util import ScriptTool
from src.data.crawler.announcement.agent import AnnouncementAgent

@ScriptTool("test_announcement_race_mode")
def main(
    start: int = 20260406,
    end: int = 20260407,
    workers: int = 10,
    race_ratio: float = 1.0,
    min_race_tasks: int = 999,
    max_replicas_per_task: int = 5,
    max_total_inflight_per_exchange: int = 20,
    **kwargs,
):
    Logger.note(
        f"Race test start={start} end={end} workers={workers} "
        f"race_ratio={race_ratio} min_race_tasks={min_race_tasks} "
        f"max_replicas_per_task={max_replicas_per_task} "
        f"max_total_inflight_per_exchange={max_total_inflight_per_exchange}"
    )
    ok = AnnouncementAgent.run_with_proxy_async(
        start=start,
        end=end,
        workers=workers,
        race_ratio=race_ratio,
        min_race_tasks=min_race_tasks,
        max_replicas_per_task=max_replicas_per_task,
        max_total_inflight_per_exchange=max_total_inflight_per_exchange,
        go_with_cached_proxies=True,
        redownload=False,
    )
    if ok:
        Logger.success("Race mode test finished successfully")
    else:
        Logger.alert1("Race mode test finished with failures")


if __name__ == "__main__":
    main()
