from typing import Literal , Any

from .fetcher import FetcherTask , EXCHANGE_URLS
from src.proj import Logger , CALENDAR , Proj , Dates
from src.proj.util.proxy import ProxyAPI , ProxyVerifier

__all__ = ["AnnouncementAgent"]

class AnnouncementAgent:
    START_DATE = 20220101

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
        vb_level = Proj.vb.level(vb_level)
        if update_type == 'recalc':
            raise ValueError(f'Recalculate all is not supported for {cls.__name__}')
        elif update_type == 'update':
            start , end , redownload = int(cls.START_DATE) , int(CALENDAR.update_to()) , False
        elif update_type == 'rollback':
            rollback_date = CALENDAR.td(cls._rollback_date)
            start , end , redownload = int(rollback_date) , int(CALENDAR.update_to()) , True
        else:
            raise ValueError(f'Invalid update type: {update_type}')
        
        success =cls.run_with_proxy(start, end, redownload = redownload , workers = workers, fallback_to_raw_ip = True, indent = indent, vb_level = vb_level, **kwargs)
        if success:
            Logger.success(f'{cls.__name__} Update at {Dates(start, end)}' , indent = indent , vb_level = vb_level)
        else:
            Logger.alert1(f'{cls.__name__} Update at {Dates(start, end)} failed')

    @classmethod
    def prepare_proxies(cls , vb_level : Any = 1):
        """verify the proxies"""
        with Logger.Paragraph(f"Prepare Proxies", level = 1, vb_level = vb_level):
            for url in EXCHANGE_URLS.values():
                ProxyAPI.get_working_proxies(url, timeout=8.0,  workers=50)
        Logger.display(ProxyVerifier.stats() , vb_level = vb_level)

    @classmethod
    def get_proxy_pool(cls , go_with_cached_proxies , * , indent : int = 1 , vb_level : Any = 1):
        """get the ProxyPool(AutoRefreshProxyPool)"""
        with Logger.Timer(f"Warmup ProxyPool", indent = indent, vb_level = vb_level) as timer:
            proxy_pool = ProxyAPI.get_proxy_pool(list(EXCHANGE_URLS.values()) , go_with_cached_proxies=go_with_cached_proxies)
            timer.add_key_suffix(f" found proxies {proxy_pool.num_proxies}")
        return proxy_pool

    @classmethod
    def run_sequential(cls , start: int, end: int, step: int = 1, redownload: bool = False , * , 
                       no_proxy: bool = False, go_with_cached_proxies : bool = False, indent : int = 1 , vb_level : Any = 1) -> bool:
        """sequential run all announcement tasks"""
        vb_level = Proj.vb.level(vb_level)
        tasks = FetcherTask.tasks_flat(start, end, step, redownload)
        Logger.stdout(f"Task iteration: total {len(tasks)} tasks (non-overlapping date blocks x exchanges)")
        if no_proxy:
            results = [False for _ in range(len(tasks))]
            for i , task in enumerate(tasks):
                results[i] = task.claw(None)
        else:
            proxy_pool = cls.get_proxy_pool(go_with_cached_proxies , indent = indent, vb_level = vb_level)
            calls = [(task.url , task.claw) for i , task in enumerate(tasks)]
            results = proxy_pool.execute(calls , max_workers=1)
        return all(results)

    @classmethod
    def run_with_proxy(cls , start: int, end: int, step: int = 1, redownload: bool = False , * , go_with_cached_proxies: bool = False,
                       workers: int = 10, group_num: int = 100, fallback_to_raw_ip : bool = False, indent : int = 1 , vb_level : Any = 1) -> bool:
        """parallel run all announcement tasks"""
        vb_level = Proj.vb.level(vb_level)
        workers = min(max(1, workers), 6)
        tasks = FetcherTask.tasks_flat(start, end, step, redownload)
        Logger.stdout(f"Task iteration: total {len(tasks)} tasks (non-overlapping date blocks x exchanges)")
        proxy_pool = cls.get_proxy_pool(go_with_cached_proxies , indent = indent, vb_level = vb_level)
        calls = [(task.url , task.claw) for i , task in enumerate(tasks)]
        results = proxy_pool.execute(calls , max_workers=workers , grouping_num=group_num , fallback_to_raw_ip=fallback_to_raw_ip)
        return all(results)

    