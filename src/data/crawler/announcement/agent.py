from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal , Any

from .fetcher import FetcherTask , ExchangeFetcherStates
from src.proj import Logger , CALENDAR , Proj , Dates

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
        
        success = cls.parallel_run(start, end, redownload = redownload , workers = workers, indent = indent, vb_level = vb_level, **kwargs)
        if success:
            Logger.success(f'{cls.__name__} Update at {Dates(start, end)}' , indent = indent , vb_level = vb_level)
        else:
            Logger.alert1(f'{cls.__name__} Update at {Dates(start, end)} failed')

    @classmethod
    def sequential_run(cls , start: int, end: int, step: int = 3, redownload: bool = False , * , indent : int = 1 , vb_level : Any = 1 , auto_discover_proxy: bool = True) -> bool:
        """sequential run all announcement tasks"""
        vb_level = Proj.vb.level(vb_level)
        cls.warmup(auto_discover_proxy, indent = indent, vb_level = vb_level)
        tasks = FetcherTask.tasks_flat(start, end, step, redownload)
        Logger.stdout(f"Task iteration: total {len(tasks)} tasks (non-overlapping date blocks x exchanges)")
        for task in tasks:
            if not any(ExchangeFetcherStates.get_states().values()):
                Logger.alert2(f"All exchanges proxies are disabled, skip at sequential run")
                return False
            task.run_with_proxies(indent = indent + 1, vb_level = vb_level + 1)
        return True

    @classmethod
    def parallel_run(cls , start: int, end: int, step: int = 3, redownload: bool = False , * ,
                      workers: int = 3, max_groups: int = 100, auto_discover_proxy: bool = True, indent : int = 1 , vb_level : Any = 1) -> bool:
        """parallel run all announcement tasks"""
        vb_level = Proj.vb.level(vb_level)
        cls.warmup(auto_discover_proxy, indent = indent, vb_level = vb_level)
        workers = min(max(1, workers), 6)
        if workers == 1:
            return cls.sequential_run(start, end, step, redownload, indent = indent, vb_level = vb_level, auto_discover_proxy=auto_discover_proxy)
        groups = FetcherTask.tasks_groups(start, end, step, redownload , max_groups=max_groups , min_tasks_per_group=workers * 3)
        Logger.stdout(f"Task partitioning: total {sum(len(g) for g in groups)} tasks (non-overlapping date blocks x exchanges), divided into {len(groups)} buckets" , 
                      indent = indent, vb_level = vb_level)
        for gi, group in enumerate(groups):
            if not any(ExchangeFetcherStates.get_states().values()):
                Logger.alert2(f"All exchanges are disabled, skip at task group {gi + 1}/{len(groups)}" , indent = indent + 1, vb_level = vb_level + 1)
                return False
            Logger.stdout(f"Task group {gi + 1}/{len(groups)}, this group has {len(group)} tasks" , indent = indent + 1, vb_level = vb_level + 1)
            with ThreadPoolExecutor(max_workers=min(workers, len(group))) as pool:
                futures = {pool.submit(task.run_with_proxies, indent = indent + 2, vb_level = vb_level + 4): task for task in group}
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:  # noqa: BLE001
                        Logger.error(f"{futures[fut].title} uncaught exception: {e}")
        return True

    @classmethod
    def warmup(cls , auto_discover_proxy: bool = True , indent : int = 1 , vb_level : Any = 1):
        """warmup the fetcher states"""
        ExchangeFetcherStates.reset_states()
        if auto_discover_proxy:
            Logger.stdout("Auto-discovery public free proxy list" , indent = indent, vb_level = vb_level)
            ExchangeFetcherStates.init_all_proxies()
            Logger.stdout(ExchangeFetcherStates.list_all_proxies() , indent = indent, vb_level = vb_level)
