#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: codex
# date: 2026-04-30
# description: Minimal Async Announcement Smoke Test
# content: 最小化验证异步公告抓取路径（不落盘），按交易所各抓一个任务并输出条数
# email: False
# mode: shell
# parameters:
#   date:
#       type: int
#       desc: target date for smoke test
#       required: False
#       default: 20260331
#   proxy:
#       type: [True, False]
#       desc: whether to use one proxy from pool
#       required: False
#       default: True

import asyncio

from src.proj import Logger
from src.proj.util import ScriptTool, ProxyAPI
from src.proj.util.proxy.ppool import AdaptiveProxyPool
from src.data.crawler.announcement.fetcher import FetcherTask


@ScriptTool("test_announcement_async_minimal")
def main(date: int = 20260331, **kwargs):
    pool = ProxyAPI.get_proxy_pool('https://www.bse.cn', go_with_cached_proxies=False)
    asyncio.run(_main_async(date=date, pool=pool))


async def _main_async(date: int, pool: AdaptiveProxyPool):
    Logger.note(f"Run async announcement smoke test at date={date}")
    task = FetcherTask(exchange='bse', start=date, end=date, redownload=True)
    proxy_url = None
    picked = None
    payload: list | Exception = Exception("not-started")
    picked = pool.acquire(task.url)
    if picked is not None:
        proxy_url = picked.url
    try:
        payload = await task.fetch_payload_async(proxy_url)
        if isinstance(payload, Exception):
            Logger.alert1(f"[smoke] {task.title} failed: {payload!s}")
        else:
            Logger.success(f"[smoke] {task.title} success, rows={len(payload)}")
    finally:
        if picked is not None:
            success = isinstance(payload, list)
            pool.release(picked, success)


if __name__ == "__main__":
    main()
