"""
Orchestrator for crawling exchange disclosure announcements.

Uses rotating proxies to fetch announcement metadata from the Shanghai and
Shenzhen stock exchange websites.  Only active on CUDA servers
(``MACHINE.cuda_server``); disabled on other machines.
"""
from __future__ import annotations
import asyncio
from collections import defaultdict
from typing import Any , Iterable

from src.proj import MACHINE , Base
from src.proj.util.web.proxy import ProxyAPI , ProxyVerifier
from src.proj.util.web.proxy.caller import ProxyCallerList
from src.proj.util.web.proxy.ppool import AsyncAdaptiveProxyPool

from .const import ExchangeType
from .fetcher_task import FetcherTask
from .async_race import AsyncProxyRaceExecutor
from .util import CrawlerLogger

__all__ = ["AnnouncementAgent"]

class AnnouncementAgent(Base.BasicUpdater):
    """
    Crawl and incrementally update exchange announcement metadata.

    Uses ``FetcherTask`` for per-date HTTP fetches and ``ProxyCallerList``
    for rotating proxy management.
    """
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE, Base.UpdateType.ROLLBACK)
    START_DATE = 20160101 if MACHINE.cuda_server else 20250101

    @classmethod
    def parse_update_input(cls , *args , **kwargs) -> dict[str , Any]:
        return super().parse_update_input(*args , key='crawler_announcement' , **kwargs)

    @classmethod
    def proceed_update(
        cls , start : int , end : int , overwrite : bool , * , workers: int = 10, force_update: int = 0, **kwargs
    ) -> Base.UpdateFlag:        
        flag = cls.run_with_proxy_async(
            start, end, redownload = overwrite , 
            force_update = force_update , workers = workers, 
            fallback_to_raw_ip = False, **kwargs)
        return flag

    @classmethod
    def prepare_proxies(cls):
        """verify the proxies"""
        with cls.logger.paragraph(f"Prepare Proxies", level = 1, vb = 1):
            for url in ExchangeType.all_urls():
                ProxyAPI.get_working_proxies(url, timeout=8.0,  workers=50)
        cls.logger.display(ProxyVerifier.stats() , vb = 1)

    @classmethod
    def get_proxy_pool(cls , urls : Iterable[str] | str = ExchangeType.all_urls() , go_with_cached_proxies = False):
        """get the ProxyPool(AutoRefreshProxyPool)"""
        with cls.logger.timer(f"Warmup ProxyPool", idt = 1, vb = 1) as timer:
            proxy_pool = ProxyAPI.get_proxy_pool(urls , go_with_cached_proxies=go_with_cached_proxies)
            timer.add_key_suffix(f" found proxies {proxy_pool.num_proxies}")
            timer.set_printer('success')
        return proxy_pool

    @classmethod
    def get_async_proxy_pool(cls, urls: Iterable[str] | str = ExchangeType.all_urls(), go_with_cached_proxies: bool = False):
        with cls.logger.timer(f"Warmup AsyncProxyPool", idt = 1, vb = 1) as timer:
            if isinstance(urls, str):
                target_urls = [urls]
            elif isinstance(urls, list):
                target_urls = urls
            else:
                target_urls = list(urls)
            proxy_pool = AsyncAdaptiveProxyPool(target_urls, go_with_cached_proxies=go_with_cached_proxies)
            timer.add_key_suffix(f" found proxies {proxy_pool.num_proxies}")
            timer.set_printer('success')
        return proxy_pool

    @classmethod
    def get_proxy_caller_list(
        cls , start: int, end: int, step: int = 1, redownload: bool = False , * , force_update: int = 0,
        use_proxy = True , go_with_cached_proxies = False, ignore_proxy_threshold : int = 0 , 
    ) -> ProxyCallerList:
        tasks = FetcherTask.tasks_flat(start, end, step, redownload , force_update=force_update)
        if tasks:
            min_date = min(task.start for task in tasks)
            max_date = max(task.end for task in tasks)
            cls.logger.stdout(f"Total Announcement Clawing Tasks: {len(tasks)} at {min_date}~{max_date} for 3 exchanges" , idt = 1, vb = 1)
        else:
            return ProxyCallerList([])
        unique_urls = set([task.url for task in tasks])
        if use_proxy:
            target_urls = [url for url in unique_urls if sum(task.url == url for task in tasks) > ignore_proxy_threshold]
        else:
            target_urls = []
        proxy_pool = cls.get_proxy_pool(target_urls, go_with_cached_proxies)
        caller_list = ProxyCallerList({task.title: task.to_proxy_caller(proxy_pool) for task in tasks} , pool = proxy_pool)
        return caller_list

    @classmethod
    def run_with_proxy(
        cls , start: int, end: int, step: int = 1, redownload: bool = False , * , 
        force_update: int = 0, go_with_cached_proxies: bool = False,
        workers: int = 10, fallback_to_raw_ip : bool = False ,
        use_async: bool = False, race_ratio: float = 0.5, min_race_tasks: int = 2,
        max_replicas_per_task: int = 5, max_total_inflight_per_exchange: int = 20 , **kwargs
    ) -> Base.UpdateFlag:
        """parallel run all announcement tasks"""
        if use_async:
            return cls.run_with_proxy_async(
                start, end, step=step, redownload=redownload, force_update=force_update,
                go_with_cached_proxies=go_with_cached_proxies, workers=workers,
                fallback_to_raw_ip=fallback_to_raw_ip, 
                race_ratio=race_ratio, min_race_tasks=min_race_tasks,
                max_replicas_per_task=max_replicas_per_task,
                max_total_inflight_per_exchange=max_total_inflight_per_exchange,
            )
        caller_list = cls.get_proxy_caller_list(
            start, end, step, redownload, force_update=force_update, use_proxy = True, 
            go_with_cached_proxies = go_with_cached_proxies)
        if caller_list:
            results = caller_list.execute_with_partition(max_workers=min(max(1, workers), 50) , fallback_to_raw_ip=fallback_to_raw_ip)
            if all([result if isinstance(result, bool) else False for result in results]):
                return Base.UpdateFlag.FAILED
            else:
                return Base.UpdateFlag.SUCCESS
        else:
            return Base.UpdateFlag.SKIPPED

    @classmethod
    async def _run_with_proxy_async(
        cls,
        start: int,
        end: int,
        step: int = 1,
        redownload: bool = False,
        *,
        force_update: int = 0,
        go_with_cached_proxies: bool = False,
        workers: int = 10,
        race_ratio: float = 0.5,
        min_race_tasks: int = 2,
        max_replicas_per_task: int = 5,
        max_total_inflight_per_exchange: int = 20,
        **kwargs,
    ) -> Base.UpdateFlag:
        tasks = FetcherTask.tasks_flat(start, end, step, redownload , force_update=force_update)
        if tasks:
            min_date = min(task.start for task in tasks)
            max_date = max(task.end for task in tasks)
            cls.logger.stdout(f"Total Announcement Clawing Tasks: {len(tasks)} at {min_date}~{max_date} for 3 exchanges" , idt = 1, vb = 1)
        else:
            return Base.UpdateFlag.SKIPPED
        grouped_tasks: dict[str, list[FetcherTask]] = defaultdict(list)
        for task in tasks:
            grouped_tasks[task.exchange].append(task)
        target_urls = sorted({task.url for task in tasks})
        proxy_pool = cls.get_async_proxy_pool(target_urls, go_with_cached_proxies=go_with_cached_proxies)
        all_ok = True

        async def run_one_exchange(exchange: str, ex_tasks: list[FetcherTask]):
            executor = AsyncProxyRaceExecutor(
                proxy_pool,
                race_ratio=race_ratio,
                min_race_tasks=min_race_tasks,
                max_replicas_per_task=max_replicas_per_task,
                max_total_inflight_per_exchange=max_total_inflight_per_exchange,
            )
            CrawlerLogger.stdout(f"[async-race] Running {exchange} with {len(ex_tasks)} tasks and {workers} workers")
            ex_result = await executor.run_exchange_tasks(ex_tasks, workers=min(max(1, workers), 50))
            for task in ex_tasks:
                payload = ex_result["results"].get(task.title)
                if payload is None:
                    continue
                task_key = f"{task.start}_{task.end}_{task.exchange}"
                task.persist_payload(payload)
                winner_attempt = ex_result.get("winner_attempt", {}).get(task.title)
                task.exporter.cleanup_temp_attempts(task_key)
                CrawlerLogger.success(
                    f"[crawler-task-finished] task={task.title} persisted_rows={len(payload)} winner={winner_attempt}",
                )
            if ex_result["errors"]:
                CrawlerLogger.alert(f"{exchange} async crawl has {len(ex_result['errors'])} failed tasks")
            return ex_result["ok"]

        exchange_results = await asyncio.gather(*[
            run_one_exchange(exchange, ex_tasks)
            for exchange, ex_tasks in grouped_tasks.items()
        ])
        all_ok = all(exchange_results)
        return Base.UpdateFlag.SUCCESS if all_ok else Base.UpdateFlag.FAILED

    @classmethod
    def run_with_proxy_async(
        cls,
        start: int,
        end: int,
        step: int = 1,
        redownload: bool = False,
        *,
        force_update: int = 0,
        go_with_cached_proxies: bool = False,
        workers: int = 10,
        fallback_to_raw_ip: bool = False,
        race_ratio: float = 0.5,
        min_race_tasks: int = 2,
        max_replicas_per_task: int = 5,
        max_total_inflight_per_exchange: int = 20,
        **kwargs,
    ) -> Base.UpdateFlag:
        if fallback_to_raw_ip:
            cls.logger.alert1("fallback_to_raw_ip is ignored in async mode")
        return asyncio.run(
            cls._run_with_proxy_async(
                start, end, step=step, redownload=redownload, force_update=force_update,
                go_with_cached_proxies=go_with_cached_proxies, workers=workers,
                race_ratio=race_ratio, min_race_tasks=min_race_tasks,
                max_replicas_per_task=max_replicas_per_task,
                max_total_inflight_per_exchange=max_total_inflight_per_exchange,
                **kwargs,
            )
        )

    