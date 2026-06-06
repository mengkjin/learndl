"""
Orchestrator for crawling exchange disclosure announcements.

Uses rotating proxies to fetch announcement metadata from the Shanghai and
Shenzhen stock exchange websites.  Only active on CUDA servers
(``MACHINE.cuda_server``); disabled on other machines.
"""
from __future__ import annotations
import asyncio
from collections import defaultdict
from typing import Literal , Any , Iterable


from src.proj import CALENDAR , MACHINE , Base
from src.proj.util.web.proxy import ProxyAPI , ProxyVerifier
from src.proj.util.web.proxy.caller import ProxyCallerList
from src.proj.util.web.proxy.ppool import AsyncAdaptiveProxyPool

from . import const
from .fetcher_task import FetcherTask
from .async_race import AsyncProxyRaceExecutor
from .util import CrawlerLogger

__all__ = ["AnnouncementAgent"]

class AnnouncementAgent(Base.BoundLogger):
    """
    Crawl and incrementally update exchange announcement metadata.

    Uses ``FetcherTask`` for per-date HTTP fetches and ``ProxyCallerList``
    for rotating proxy management.
    """
    START_DATE = 20160101 if MACHINE.cuda_server else 20260101

    @classmethod
    def update(cls):
        cls.update_all('update')

    @classmethod
    def rollback(cls , rollback_date : int):
        cls.set_rollback_date(rollback_date)
        cls.update_all('rollback')

    @classmethod
    def recalculate_all(cls):
        cls.update_all('recalc')

    @classmethod
    def set_rollback_date(cls , rollback_date : int):
        CALENDAR.check_rollback_date(rollback_date)
        cls._rollback_date = rollback_date

    @classmethod
    def update_all(
        cls , update_type : Literal['recalc' , 'update' , 'rollback'] , * , 
        indent : int = 0 , vb_level : Any = 1 , workers: int = 10, force_update: int = 0, **kwargs
    ):
        cls.SetClassVB(vb_level , indent)
        if update_type == 'recalc':
            cls.logger.note(f'Recalculate all!')
            raise ValueError(f'Recalculate all is not supported for {cls.__name__}')
        elif update_type == 'update':
            cls.logger.note(f'Update since last update!')
            start , end , redownload = int(cls.START_DATE) , int(CALENDAR.update_to()) , False
        elif update_type == 'rollback':
            cls.logger.note(f'Rollback from {cls._rollback_date}!')
            rollback_date = CALENDAR.td(cls._rollback_date)
            start , end , redownload = int(rollback_date) , int(CALENDAR.update_to()) , True
        else:
            raise ValueError(f'Invalid update type: {update_type}')
        
        status = cls.run_with_proxy_async(start, end, redownload = redownload , force_update = force_update , workers = workers, fallback_to_raw_ip = False, **kwargs)
        if status == 'skipping':
            cls.logger.skipping(f'Announcements have already been updated recently' , idt = 1)
        elif status == 'success':
            cls.logger.success(f'Announcements updated at {Base.Dates(end)}' , idt = 1)
        else:
            cls.logger.alert1(f'Announcements update at {Base.Dates(end)} failed' , idt = 1)

    @classmethod
    def prepare_proxies(cls):
        """verify the proxies"""
        with cls.logger.paragraph(f"Prepare Proxies", level = 1, vb = 1):
            for url in const.EXCHANGE_URLS.values():
                ProxyAPI.get_working_proxies(url, timeout=8.0,  workers=50)
        cls.logger.display(ProxyVerifier.stats() , vb = 1)

    @classmethod
    def get_proxy_pool(cls , urls : Iterable[str] | str = const.EXCHANGES , go_with_cached_proxies = False):
        """get the ProxyPool(AutoRefreshProxyPool)"""
        with cls.logger.timer(f"Warmup ProxyPool", idt = 1, vb = 1) as timer:
            proxy_pool = ProxyAPI.get_proxy_pool(urls , go_with_cached_proxies=go_with_cached_proxies)
            timer.add_key_suffix(f" found proxies {proxy_pool.num_proxies}")
            timer.set_printer('success')
        return proxy_pool

    @classmethod
    def get_async_proxy_pool(cls, urls: Iterable[str] | str = const.EXCHANGES, go_with_cached_proxies: bool = False):
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
        max_replicas_per_task: int = 5, max_total_inflight_per_exchange: int = 20
    ) -> Literal['skipping' , 'success' , 'failed']:
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
            return 'success' if all([result if isinstance(result, bool) else False for result in results]) else 'failed'
        else:
            return 'skipping'

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
    ) -> Literal['skipping' , 'success' , 'failed']:
        tasks = FetcherTask.tasks_flat(start, end, step, redownload , force_update=force_update)
        if tasks:
            min_date = min(task.start for task in tasks)
            max_date = max(task.end for task in tasks)
            cls.logger.stdout(f"Total Announcement Clawing Tasks: {len(tasks)} at {min_date}~{max_date} for 3 exchanges" , idt = 1, vb = 1)
        else:
            return 'skipping'
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
        return 'success' if all_ok else 'failed'

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
    ) -> Literal['skipping' , 'success' , 'failed']:
        if fallback_to_raw_ip:
            cls.logger.alert1("fallback_to_raw_ip is ignored in async mode")
        return asyncio.run(
            cls._run_with_proxy_async(
                start, end, step=step, redownload=redownload, force_update=force_update,
                go_with_cached_proxies=go_with_cached_proxies, workers=workers,
                race_ratio=race_ratio, min_race_tasks=min_race_tasks,
                max_replicas_per_task=max_replicas_per_task,
                max_total_inflight_per_exchange=max_total_inflight_per_exchange,
            )
        )

    