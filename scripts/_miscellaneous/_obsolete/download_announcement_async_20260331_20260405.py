#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: codex
# date: 2026-04-30
# description: Download Announcement Async By Date Range
# content: 使用异步公告调度下载 20260331-20260405 区间公告，支持代理竞速参数
# email: True
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
#       default: 20260405
#   workers:
#       type: int
#       desc: max workers for async scheduler
#       required: False
#       default: 10
#   race_ratio:
#       type: float
#       desc: switch to race mode when remaining tasks below threshold ratio
#       required: False
#       default: 0.5
#   min_race_tasks:
#       type: int
#       desc: minimum threshold cap for race mode
#       required: False
#       default: 2
#   max_replicas_per_task:
#       type: int
#       desc: max race replicas for one task
#       required: False
#       default: 5
#   max_total_inflight_per_exchange:
#       type: int
#       desc: max inflight tasks per exchange
#       required: False
#       default: 20
#   go_with_cached_proxies:
#       type: [True, False]
#       desc: use cached proxies when warming pool
#       required: False
#       default: True
#   redownload:
#       type: [True, False]
#       desc: force redownload even if files exist
#       required: False
#       default: False

from src.proj import Logger
from src.proj.util.script import ScriptTool
from src.data.crawler.announcement.agent import AnnouncementAgent


@ScriptTool("download_announcement_async_20260331_20260405")
def main(
    start: int = 20260331,
    end: int = 20260405,
    workers: int = 10,
    race_ratio: float = 0.5,
    min_race_tasks: int = 2,
    max_replicas_per_task: int = 5,
    max_total_inflight_per_exchange: int = 20,
    go_with_cached_proxies: bool = True,
    redownload: bool = False,
    **kwargs,
):
    Logger.note(
        f"Run async announcement download: {start}~{end}, workers={workers}, "
        f"race_ratio={race_ratio}, min_race_tasks={min_race_tasks}, "
        f"max_replicas_per_task={max_replicas_per_task}, "
        f"max_total_inflight_per_exchange={max_total_inflight_per_exchange}"
    )
    ok = AnnouncementAgent.run_with_proxy_async(
        start=start,
        end=end,
        step=1,
        redownload=redownload,
        go_with_cached_proxies=go_with_cached_proxies,
        workers=workers,
        race_ratio=race_ratio,
        min_race_tasks=min_race_tasks,
        max_replicas_per_task=max_replicas_per_task,
        max_total_inflight_per_exchange=max_total_inflight_per_exchange,
    )
    if ok:
        Logger.success(f"Async announcement download completed: {start}~{end}")
    else:
        Logger.alert1(f"Async announcement download finished with failures: {start}~{end}")


if __name__ == "__main__":
    main()
