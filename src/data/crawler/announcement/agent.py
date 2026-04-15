"""
Orchestrator for crawling exchange disclosure announcements.

Uses rotating proxies to fetch announcement metadata from the Shanghai and
Shenzhen stock exchange websites.  Only active on CUDA servers
(``MACHINE.cuda_server``); disabled on other machines.
"""
from typing import Literal , Any , Iterable

from .fetcher import FetcherTask , EXCHANGE_URLS
from src.proj import Logger , CALENDAR , Proj , Dates , MACHINE
from src.proj.util.proxy import ProxyAPI , ProxyVerifier
from src.proj.util.proxy.caller import ProxyCallerList

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
    def update_all(cls , update_type : Literal['recalc' , 'update' , 'rollback'] , * , indent : int = 1 , vb_level : Any = 1 , workers: int = 3, **kwargs):
        vb_level = Proj.vb(vb_level)
        if update_type == 'recalc':
            raise ValueError(f'Recalculate all is not supported for {cls.__name__}')
        elif update_type == 'update':
            start , end , redownload = int(cls.START_DATE) , int(CALENDAR.update_to()) , False
        elif update_type == 'rollback':
            rollback_date = CALENDAR.td(cls._rollback_date)
            start , end , redownload = int(rollback_date) , int(CALENDAR.update_to()) , True
        else:
            raise ValueError(f'Invalid update type: {update_type}')
        
        success = cls.run_with_proxy(start, end, redownload = redownload , workers = workers, fallback_to_raw_ip = False, indent = indent, vb_level = vb_level, **kwargs)
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
    def get_proxy_caller_list(
        cls , start: int, end: int, step: int = 1, redownload: bool = False , * ,
        use_proxy = True , go_with_cached_proxies = False, ignore_proxy_threshold : int = 0 , 
        indent : int = 1 , vb_level : Any = 1
    ) -> ProxyCallerList:
        tasks = FetcherTask.tasks_flat(start, end, step, redownload)
        min_date = min(task.start for task in tasks)
        max_date = max(task.end for task in tasks)
        Logger.stdout(f"Total Announcement Clawing Tasks: {len(tasks)} at {min_date}~{max_date} for 3 exchanges" , indent = indent, vb_level = vb_level)
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
                       workers: int = 10, fallback_to_raw_ip : bool = False , indent : int = 0 , vb_level : Any = 1) -> bool:
        """parallel run all announcement tasks"""
        vb_level = Proj.vb(vb_level)
        caller_list = cls.get_proxy_caller_list(
            start, end, step, redownload, use_proxy = True, 
            go_with_cached_proxies = go_with_cached_proxies , indent = indent, vb_level = vb_level)
        results = caller_list.execute_with_partition(max_workers=min(max(1, workers), 50) , fallback_to_raw_ip=fallback_to_raw_ip)
        return all([result if isinstance(result, bool) else False for result in results])

    