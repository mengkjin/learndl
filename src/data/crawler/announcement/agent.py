"""
Orchestrator for crawling exchange disclosure announcements.

Uses rotating proxies to fetch announcement metadata from the Shanghai and
Shenzhen stock exchange websites.  Only active on CUDA servers
(``MACHINE.cuda_server``); disabled on other machines.
"""
import asyncio
from collections import defaultdict
from typing import Literal , Any , Iterable


from src.proj import Logger , CALENDAR , Proj , Dates , MACHINE
from src.proj.util.proxy import ProxyAPI , ProxyVerifier
from src.proj.util.proxy.caller import ProxyCallerList
from src.proj.util.proxy.ppool import AsyncAdaptiveProxyPool

from .fetcher import FetcherTask , EXCHANGE_URLS
from .async_race import AsyncProxyRaceExecutor
from .util import fetch_log

__all__ = ["AnnouncementAgent"]

class AnnouncementAgent:
    """
    Crawl and incrementally update exchange announcement metadata.

    Uses ``FetcherTask`` for per-date HTTP fetches and ``ProxyCallerList``
    for rotating proxy management.
    """
    START_DATE = 20160101 if MACHINE.cuda_server else 20260101

    @classmethod
    def update(cls):
        Logger.note(f'Update: {cls.__name__} since last update!')
        cls.update_all('update')

    @classmethod
    def rollback(cls , rollback_date : int):
        Logger.note(f'Update: {cls.__name__} rollback from {rollback_date}!')
        cls.set_rollback_date(rollback_date)
        cls.update_all('rollback')

    @classmethod
    def recalculate_all(cls):
        Logger.note(f'Update: {cls.__name__} recalculate all!')
        cls.update_all('recalc')

    @classmethod
    def set_rollback_date(cls , rollback_date : int):
        CALENDAR.check_rollback_date(rollback_date)
        cls._rollback_date = rollback_date

    @classmethod
    def update_all(
        cls , update_type : Literal['recalc' , 'update' , 'rollback'] , * , 
        indent : int = 1 , vb_level : Any = 1 , workers: int = 10, redownload: bool = False, **kwargs
    ):
        vb_level = Proj.vb(vb_level)
        if update_type == 'recalc':
            raise ValueError(f'Recalculate all is not supported for {cls.__name__}')
        elif update_type == 'update':
            start , end , redownload = int(cls.START_DATE) , int(CALENDAR.update_to()) , False or redownload
        elif update_type == 'rollback':
            rollback_date = CALENDAR.td(cls._rollback_date)
            start , end , redownload = int(rollback_date) , int(CALENDAR.update_to()) , True or redownload
        else:
            raise ValueError(f'Invalid update type: {update_type}')
        
        success = cls.run_with_proxy_async(start, end, redownload = redownload , workers = workers, fallback_to_raw_ip = False, indent = indent, vb_level = vb_level, **kwargs)
        if success:
            Logger.success(f'{cls.__name__} Update at {Dates(end)}' , indent = indent , vb_level = vb_level)
        else:
            Logger.alert1(f'{cls.__name__} Update at {Dates(end)} failed')

    @classmethod
    def prepare_proxies(cls , vb_level : Any = 1):
        """verify the proxies"""
        with Logger.Paragraph(f"Prepare Proxies", level = 1, vb_level = vb_level):
            for url in EXCHANGE_URLS.values():
                ProxyAPI.get_working_proxies(url, timeout=8.0,  workers=50)
        Logger.display(ProxyVerifier.stats() , vb_level = vb_level)

    @classmethod
    def get_proxy_pool(cls , urls : Iterable[str] | str = EXCHANGE_URLS.keys() , go_with_cached_proxies = False , * , indent : int = 1 , vb_level : Any = 1):
        """get the ProxyPool(AutoRefreshProxyPool)"""
        with Logger.Timer(f"Warmup ProxyPool", indent = indent, vb_level = vb_level) as timer:
            proxy_pool = ProxyAPI.get_proxy_pool(urls , go_with_cached_proxies=go_with_cached_proxies)
            timer.add_key_suffix(f" found proxies {proxy_pool.num_proxies}")
        return proxy_pool

    @classmethod
    def get_async_proxy_pool(cls, urls: Iterable[str] | str = EXCHANGE_URLS.keys(), go_with_cached_proxies: bool = False, *, indent: int = 1, vb_level: Any = 1):
        with Logger.Timer(f"Warmup AsyncProxyPool", indent=indent, vb_level=vb_level) as timer:
            if isinstance(urls, str):
                target_urls = [urls]
            elif isinstance(urls, list):
                target_urls = urls
            else:
                target_urls = list(urls)
            proxy_pool = AsyncAdaptiveProxyPool(target_urls, go_with_cached_proxies=go_with_cached_proxies)
            timer.add_key_suffix(f" found proxies {proxy_pool.num_proxies}")
        return proxy_pool

    @classmethod
    def get_proxy_caller_list(
        cls , start: int, end: int, step: int = 1, redownload: bool = False , * ,
        use_proxy = True , go_with_cached_proxies = False, ignore_proxy_threshold : int = 0 , 
        indent : int = 1 , vb_level : Any = 1
    ) -> ProxyCallerList:
        tasks = FetcherTask.tasks_flat(start, end, step, redownload)
        if tasks:
            min_date = min(task.start for task in tasks)
            max_date = max(task.end for task in tasks)
            Logger.stdout(f"Total Announcement Clawing Tasks: {len(tasks)} at {min_date}~{max_date} for 3 exchanges" , indent = indent, vb_level = vb_level)
        else:
            return ProxyCallerList([])
        unique_urls = set([task.url for task in tasks])
        if use_proxy:
            target_urls = [url for url in unique_urls if sum(task.url == url for task in tasks) > ignore_proxy_threshold]
        else:
            target_urls = []
        proxy_pool = cls.get_proxy_pool(target_urls, go_with_cached_proxies, indent = indent, vb_level = vb_level)
        caller_list = ProxyCallerList({task.title: task.to_proxy_caller(proxy_pool) for task in tasks} , pool = proxy_pool)
        return caller_list

    @classmethod
    def run_with_proxy(cls , start: int, end: int, step: int = 1, redownload: bool = False , * , go_with_cached_proxies: bool = False,
                       workers: int = 10, fallback_to_raw_ip : bool = False , indent : int = 0 , vb_level : Any = 1,
                       use_async: bool = False, race_ratio: float = 0.5, min_race_tasks: int = 2,
                       max_replicas_per_task: int = 5, max_total_inflight_per_exchange: int = 20) -> bool:
        """parallel run all announcement tasks"""
        vb_level = Proj.vb(vb_level)
        if use_async:
            return cls.run_with_proxy_async(
                start, end, step=step, redownload=redownload,
                go_with_cached_proxies=go_with_cached_proxies, workers=workers,
                fallback_to_raw_ip=fallback_to_raw_ip, indent=indent, vb_level=vb_level,
                race_ratio=race_ratio, min_race_tasks=min_race_tasks,
                max_replicas_per_task=max_replicas_per_task,
                max_total_inflight_per_exchange=max_total_inflight_per_exchange,
            )
        caller_list = cls.get_proxy_caller_list(
            start, end, step, redownload, use_proxy = True, 
            go_with_cached_proxies = go_with_cached_proxies , indent = indent, vb_level = vb_level)
        if caller_list:
            results = caller_list.execute_with_partition(max_workers=min(max(1, workers), 50) , fallback_to_raw_ip=fallback_to_raw_ip)
            return all([result if isinstance(result, bool) else False for result in results])
        else:
            return True

    @classmethod
    async def _run_with_proxy_async(
        cls,
        start: int,
        end: int,
        step: int = 1,
        redownload: bool = False,
        *,
        go_with_cached_proxies: bool = False,
        workers: int = 10,
        indent: int = 0,
        vb_level: Any = 1,
        race_ratio: float = 0.5,
        min_race_tasks: int = 2,
        max_replicas_per_task: int = 5,
        max_total_inflight_per_exchange: int = 20,
    ) -> bool:
        vb_level = Proj.vb(vb_level)
        tasks = FetcherTask.tasks_flat(start, end, step, redownload)
        if not tasks:
            return True
        grouped_tasks: dict[str, list[FetcherTask]] = defaultdict(list)
        for task in tasks:
            grouped_tasks[task.exchange].append(task)
        target_urls = sorted({task.url for task in tasks})
        proxy_pool = cls.get_async_proxy_pool(
            target_urls, go_with_cached_proxies=go_with_cached_proxies, indent=indent, vb_level=vb_level
        )
        all_ok = True

        async def run_one_exchange(exchange: str, ex_tasks: list[FetcherTask]):
            executor = AsyncProxyRaceExecutor(
                proxy_pool,
                race_ratio=race_ratio,
                min_race_tasks=min_race_tasks,
                max_replicas_per_task=max_replicas_per_task,
                max_total_inflight_per_exchange=max_total_inflight_per_exchange,
            )
            fetch_log(f"[async-race] Running {exchange} with {len(ex_tasks)} tasks and {workers} workers", type='stdout')
            ex_result = await executor.run_exchange_tasks(ex_tasks, workers=min(max(1, workers), 50))
            for task in ex_tasks:
                payload = ex_result["results"].get(task.title)
                if payload is None:
                    continue
                task_key = f"{task.start}_{task.end}_{task.exchange}"
                task.persist_payload(payload)
                winner_attempt = ex_result.get("winner_attempt", {}).get(task.title)
                task.exporter.cleanup_temp_attempts(task_key)
                fetch_log(
                    f"[crawler-task-finished] task={task.title} persisted_rows={len(payload)} winner={winner_attempt}",
                    type='success'
                )
            if ex_result["errors"]:
                fetch_log(f"{exchange} async crawl has {len(ex_result['errors'])} failed tasks", type='alert')
            return ex_result["ok"]

        exchange_results = await asyncio.gather(*[
            run_one_exchange(exchange, ex_tasks)
            for exchange, ex_tasks in grouped_tasks.items()
        ])
        all_ok = all(exchange_results)
        return all_ok

    @classmethod
    def run_with_proxy_async(
        cls,
        start: int,
        end: int,
        step: int = 1,
        redownload: bool = False,
        *,
        go_with_cached_proxies: bool = False,
        workers: int = 10,
        fallback_to_raw_ip: bool = False,
        indent: int = 0,
        vb_level: Any = 1,
        race_ratio: float = 0.5,
        min_race_tasks: int = 2,
        max_replicas_per_task: int = 5,
        max_total_inflight_per_exchange: int = 20,
    ) -> bool:
        if fallback_to_raw_ip:
            Logger.alert1("fallback_to_raw_ip is ignored in async mode")
        return asyncio.run(
            cls._run_with_proxy_async(
                start, end, step=step, redownload=redownload,
                go_with_cached_proxies=go_with_cached_proxies, workers=workers,
                indent=indent, vb_level=vb_level, race_ratio=race_ratio, min_race_tasks=min_race_tasks,
                max_replicas_per_task=max_replicas_per_task,
                max_total_inflight_per_exchange=max_total_inflight_per_exchange,
            )
        )

    