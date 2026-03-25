import random
import threading
import time
import numpy as np
from datetime import datetime , timedelta
from typing import Callable , Any , Iterable , Literal , Union, Iterator

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.proj.log import Logger
from src.proj.util.http import iterate_with_interval_control
from .abc import ProxySet , ProxyStats , ProxyStatsSet
from .verifier import ProxyVerifier
from .finder import FreeProxyFinder as ProxyFinder
from .cache import ProxyCache

ProxyCallerInput = Union[Callable[..., bool | Exception] , tuple[str, Callable[..., bool | Exception]] , tuple[Callable[..., bool | Exception], str] , 'ProxyCaller']

def test_proxies():
    """Test fake proxies"""
    return ["http://1.2.3.4:8080", "http://5.6.7.8:3128", "http://9.10.11.12:9999", "http://4.4.4.4:8080", "http://5.5.5.5:3129"]

def get_working_proxies(
    target_url: str , min_count: int = 5, * , max_round: int = 3 , timeout: float = 10.0, workers: int = 50 , 
    dummy: bool = False, go_with_cached_proxies: bool = False, detail_level: Literal['all','none','simple'] = 'all') -> ProxySet:
    """
    return the list of available proxy URLs; with in-process short-term cache to avoid hitting the free proxy site for each failed task.
    if force_refresh is True, ignore the expired cache.
    """
    Logger.highlight(f'Get working proxies for {target_url}' + (f' with dummy verification' if dummy else '') , vb_level = 0 if detail_level in ['all' , 'simple'] else 'max')
    finder = ProxyFinder()
    for _ in range(2):
        verified_proxies = ProxySet()
        for round in range(max_round + 1):
            prefix = f'Round {round} '
            if len(verified_proxies) >= min_count:
                break
            timer_find = Logger.Timer(f'{prefix}{"Load" if round == 0 else "Find"} Proxies', indent = 1, vb_level = 1 if detail_level == 'all' else 'max')
            timer_quick = Logger.Timer(f'{prefix}Quick Verify ({timeout/2:.1f}s) for {ProxyVerifier.QUICK_VERIFY_URL}', indent = 1, vb_level = 1 if detail_level == 'all' else 'max')
            timer_final = Logger.Timer(f'{prefix}Final Verify ({timeout:.1f}s) for {target_url}', indent = 1, vb_level = 1 if detail_level == 'all' else 'max')
            with timer_find as timer:
                cands = ProxyCache.get_cached_proxies('all') if round == 0 else finder.find()
                timer.add_key_suffix(f', get {len(cands)} new proxies')
            if go_with_cached_proxies and round == 0:
                return cands
            else:
                cands = ProxyVerifier.unverified_proxies(target_url, cands)
            if not cands:
                continue
            with timer_quick as timer:
                passed_proxies = ProxyVerifier.parallel_verification(cands,ProxyVerifier.QUICK_VERIFY_URL,timeout,fast_test=True,workers=workers,dummy=dummy)
                timer.add_key_suffix(f', {len(passed_proxies)}/{len(cands)} passed')
            if not passed_proxies:
                continue
            with timer_final as timer:
                final_proxies = ProxyVerifier.parallel_verification(passed_proxies,target_url,timeout,fast_test=False,workers=workers,dummy=dummy)
                timer.add_key_suffix(f', {len(final_proxies)}/{len(passed_proxies)} passed')
            verified_proxies.extend(final_proxies)
        if verified_proxies:
            break
    if not dummy and verified_proxies:
        ProxyCache.update(target_url, verified_proxies)
    return verified_proxies

class ProxyDepletionException(Exception):
    """Exception raised when the proxy pool is depleted"""
    pass

class ProxyCaller:
    """A function that can be used to execute a function with a proxy"""
    def __init__(self, pool: 'ProxyPool', func: Callable[..., bool | Exception] , url: str = ''):
        self.pool = pool
        self.func = func
        self.url = url
        self.finished = False
        self.result = False
        self.banned = False

    def __call__(self, *args: Any, **kwargs: Any) -> bool | Exception:
        return self.proxied(*args, **kwargs)

    def set_pool(self , pool: 'ProxyPool') -> 'ProxyCaller':
        self.pool = pool
        return self

    def ban(self):
        """Ban the caller"""
        self.banned = True

    def proxied(self , *args: Any, **kwargs: Any) -> bool | Exception:
        """Call the function with a proxy"""
        proxy = self.pool.acquire(self.url)
        if proxy is None:
            return ProxyDepletionException(self.url)
        self.result = self.func(proxy.url , *args, **kwargs)
        self.finished = not isinstance(self.result, Exception)
        self.pool.release(proxy, False if isinstance(self.result, Exception) else self.result)
        return self.result

    def fallback(self , *args: Any, **kwargs: Any) -> bool:
        """Fallback to raw ip"""
        self.result = self.func(None , *args, **kwargs)
        self.finished = True
        return False if isinstance(self.result, Exception) else self.result

    @classmethod
    def from_input(cls , input: ProxyCallerInput, pool: 'ProxyPool | None' = None) -> 'ProxyCaller':
        if isinstance(input , ProxyCaller):
            if pool is not None:
                input.pool = pool
            return input
        else:
            assert pool is not None , "pool is required"
            if isinstance(input , tuple):
                assert len(input) == 2 , f"input should be a tuple of length 2, contains url and func, but got {input}"
                if isinstance(input[0], str) and isinstance(input[1], Callable):
                    return cls(pool , input[1] , input[0])
                elif isinstance(input[0], Callable) and isinstance(input[1], str):
                    return cls(pool , input[0] , input[1])
                else:
                    raise ValueError(f"Invalid input: {input}")
            else:
                assert isinstance(input , Callable) , f"single input should be a callable, but got {input}"
                return cls(pool , input)

class ProxyCallerList:
    """A list of proxy callers"""
    fallback_interval = 1.0
    
    def __init__(self, callers: list['ProxyCaller'] , pool: 'ProxyPool'):
        self.callers = callers
        self.pool = pool

    def __len__(self) -> int:
        return len(self.callers)

    def __bool__(self) -> bool:
        return bool(self.callers)

    def __iter__(self) -> Iterator['ProxyCaller']:
        return iter(self.callers)

    def set_pool(self , pool: 'ProxyPool | None') -> 'ProxyCallerList':
        if pool is None:
            return self
        if pool is not self.pool:
            self.pool = pool
            [caller.set_pool(pool) for caller in self.callers]
        return self

    @property
    def all_finished(self) -> bool:
        return len(self) == 0 or all(caller.finished for caller in self.callers)

    def is_unable_to_proceed(self) -> bool:
        """Whether the proxy pool is unable to proceed"""
        self.check_shutdown()
        return len(self) == 0 or all(caller.finished or caller.banned for caller in self.callers)

    def print_status(self):
        """Print the status of the proxy pool"""
        Logger.stdout([(caller.url, caller.finished, caller.banned) for caller in self.callers])

    def check_shutdown(self):
        """Ban the url"""
        shutdown_urls = [url for url in self.pool.verify_urls if self.pool.url_shutdown(url)]
        [caller.ban() for caller in self.callers if caller.url in shutdown_urls]

    def results(self) -> list[bool | Exception]:
        return [caller.result for caller in self.callers]

    def realigned_callers(self) -> list['ProxyCaller']:
        """Realign callers to the make iteration more diverse"""
        step_size = int(np.round(np.sqrt(len(self.callers))))
        sor = sorted(self.callers , key = lambda x: x.url)
        new = []
        for i in range(0, step_size):
            new.extend(sor[i::step_size])
        assert len(new) == len(new) , (len(new) , len(self.callers))
        return new

    def execute(self , * , max_workers : int = 3):
        """Execute the unfinished callers with a thread pool"""
        unfinished_callers = [caller for caller in self.callers if not caller.finished]
        if max_workers == 1:
            for caller in unfinished_callers:
                caller.proxied()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_map = {pool.submit(caller.proxied): caller for caller in unfinished_callers}
                for fut in as_completed(fut_map):
                    fut.result()

    def fallback(self):
        """Fallback to raw ip , sleep for a while to avoid hitting the rate limit"""
        unfinished_callers = iterate_with_interval_control([caller for caller in self.callers if not caller.finished] , interval = self.fallback_interval)
        for caller in unfinished_callers:
            caller.fallback()

    @classmethod
    def from_inputs(
        cls , inputs : Iterable[ProxyCallerInput] , pool: 'ProxyPool | None' = None
    ) -> 'ProxyCallerList':
        """Create a ProxyCallerList from inputs of Iterable[func: Callable[..., bool]] | Iterable[tuple[url: str, func: Callable[..., bool]]]"""
        if isinstance(inputs , ProxyCallerList):
            return inputs.set_pool(pool)
        assert pool is not None , "pool is required"
        callers : list[ProxyCaller] = []
        for func in inputs:
            callers.append(ProxyCaller.from_input(func , pool))
        self = cls(callers , pool)
        return self

    def partition(self , max_grouping_num : int = 100) -> list['ProxyCallerList']:
        """Partition the callers into groups"""
        if not self.callers:
            return []
        callers = self.realigned_callers()
        grouping_num = max(1, min(int(np.round(np.sqrt(len(self.callers)))), max_grouping_num))
        num_groups = int(np.ceil(len(callers) / grouping_num))
        max_callers_per_group = int(np.ceil(len(callers) / num_groups))
        groups = []
        for i in range(num_groups):
            groups.append(ProxyCallerList(callers[i * max_callers_per_group:(i + 1) * max_callers_per_group] , self.pool))
        return groups

class ProxyPool:
    """Proxy pool, can be used to acquire and release proxies in a thread-safe manner"""
    def __init__(self, target_url: list[str] | str | Literal['test'], * , 
                 go_with_cached_proxies: bool = False,
                 invalid_threshold: int | None = None, max_concurrent: int | None = None):
        if target_url == 'test':
            Logger.alert1("Using test mode and pseudo proxies")
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.verify_urls = target_url if isinstance(target_url , list) else [target_url]
        ProxyStats.set_class_attrs(invalid_threshold = invalid_threshold, max_concurrent = max_concurrent)
        self.proxies : dict[str , ProxyStatsSet] = {url: ProxyStatsSet() for url in self.verify_urls}
        self.refresh_time = {url: datetime.now() for url in self.verify_urls}
        self.invalid_printed = {url: False for url in self.verify_urls}
        self.refresh_proxies(url = 'all' , go_with_cached_proxies=go_with_cached_proxies , detail_level = 'simple')
        
    def refresh_proxies(self , url : str | Literal['all' , 'test'] , go_with_cached_proxies: bool = False , detail_level: Literal['all','none','simple'] = 'all'):
        """Get the working proxies from the pool's proxy API"""
        if url == 'all':
            urls = self.verify_urls
        else:
            urls = [url]
        with self.condition:
            for url in urls:
                self.proxies[url] = ProxyStatsSet([proxy for proxy in self.proxies.get(url, []) if not proxy.invalid])
                self.proxies[url].extend(ProxyStatsSet(self.get_working_proxies(url, go_with_cached_proxies , detail_level = detail_level)))
                self.refresh_time[url] = datetime.now()

    @classmethod
    def get_working_proxies(cls , target_url : str , go_with_cached_proxies: bool = False , detail_level: Literal['all','none','simple'] = 'all') -> ProxySet:
        """Get the working proxies from the pool's proxy API"""
        if target_url == 'test':
            return ProxySet(test_proxies())
        else:
            return get_working_proxies(target_url, go_with_cached_proxies=go_with_cached_proxies , detail_level=detail_level)

    def __len__(self) -> int:
        return len(self.verify_urls)

    def __bool__(self) -> bool:
        return not self.shutdown

    @property
    def num_proxies(self) -> dict[str, int]:
        return {url: len(self.proxies[url]) for url in self.verify_urls}

    @property
    def shutdown(self) -> bool:
        """Whether the proxy pool is shutdown"""
        return all(self.url_shutdown(url) for url in self.verify_urls)

    def url_shutdown(self , url: str) -> bool:
        """Whether the url is shutdown"""
        return self.is_invalid_url(url)

    def is_invalid_url(self , url: str) -> bool:
        """Whether the url is invalid"""
        return self.proxies[url].valid_ratio == 0

    def print_status(self):
        """Print the status of the proxy pool"""
        Logger.stdout([(url, len(self.proxies[url]), self.proxies[url].valid_ratio) for url in self.verify_urls])

    def acquire(self , url: str) -> ProxyStats | None:
        """Acquire a proxy, if no available proxy but some proxies are not invalid, wait until a proxy is available, otherwise return None"""
        assert url , "url is required"
        with self.condition:
            while True:
                # condition 1: if all proxies for the url are invalid, return None
                if self.is_invalid_url(url):
                    if not self.invalid_printed[url]:
                        Logger.alert2(f"All proxies for url {url} are invalid, return None")
                        self.invalid_printed[url] = True
                    return None

                # condition 2: if there are available proxies, randomly select one and return it
                if proxy := self.proxies[url].pick_one():
                    return proxy

                # condition 3: if there are no available proxies, wait until a proxy is available
                self.condition.wait()

    def release(self, proxy: ProxyStats, success: bool) -> None:
        """Release a proxy, and update the state of this usage."""
        with self.condition:
            proxy.release(success)
            # notify all waiting threads that the proxy state has changed
            self.condition.notify_all()

    def execute(self , url_func_tuples : Iterable[ProxyCallerInput] , * , max_workers : int = 3 , grouping_num : int = 100 , **kwargs) -> list[bool | Exception]:
        callers = ProxyCallerList.from_inputs(url_func_tuples , self)
        groups = callers.partition(grouping_num)
        for group in groups:
            group.execute(max_workers=max_workers , **kwargs)
            if group.is_unable_to_proceed():
                break
        return callers.results()

    @classmethod
    def test(cls):
        """return a test proxy pool"""
        return cls('test' , invalid_threshold=1 , max_concurrent=2)

    def test_workers(self , max_workers : int = 3):
        """test a number of workers"""
        def test_worker(proxy: str, worker_id: int | None = None) -> bool:
            """A test worker thread"""
            # simulate work: random success or failure
            time.sleep(random.uniform(0.5, 2))
            success = random.choice([True, False , False , False ,False])  # random success or failure
            # success = True  # if you want all success, change to True

            if success:
                Logger.note(f"Worker {worker_id} use proxy {proxy} successfully")
            else:
                Logger.note(f"Worker {worker_id} use proxy {proxy} failed")

            return success
        Logger.stdout('start testing proxy pool')
        workers = [('test' ,lambda proxy, i=i: test_worker(proxy, worker_id=i)) for i in range(4 * len(self))]
        return self.execute(workers , max_workers=max_workers)

class AdaptiveProxyPool(ProxyPool):
    """
    Proxy pool that can be automatically refreshed with new proxies
    init args:
        target_url: the url to verify the proxies, i.e. the target url to access
        refresh_interval: the minimum interval to refresh the proxies , if all proxies down before the interval, will exit the pool
        refresh_attempts: the maximum attempts to refresh the proxies
        refresh_threshold: the ratio of initial proxies that are remained enabled to trigger a refresh
        invalid_threshold: the threshold to mark the proxies as invalid
        max_concurrent: the max concurrent proxies to use
    """
    def __init__(
        self, target_url: list[str] | str | Literal['test'] , * , 
        go_with_cached_proxies: bool = False,
        refresh_interval: int = 180 ,
        refresh_max_attempts: int = 10 ,
        refresh_threshold: float = 0.2 ,
        invalid_threshold: int = 3 , max_concurrent: int = 2
    ):
        super().__init__(
            target_url , 
            go_with_cached_proxies=go_with_cached_proxies, 
            invalid_threshold=invalid_threshold , 
            max_concurrent=max_concurrent
        )
        self.invalid_printed = {url: True for url in self.verify_urls}
        self.refresh_enabled = {url: True for url in self.verify_urls}
        self.refresh_attempt = {url: 0 for url in self.verify_urls}
        self.refresh_time    = {url: datetime.now() for url in self.verify_urls}
        self.refresh_interval = refresh_interval
        self.refresh_max_attempts = refresh_max_attempts
        self.refresh_threshold = refresh_threshold

    def url_shutdown(self , url: str) -> bool:
        """Whether the url is shutdown"""
        return self.is_invalid_url(url) and not self.refresh_enabled[url]

    def refresh(self):
        """
        Refresh the proxy pool with new proxies
        will only refresh if valid proxy ratio is less than the refresh threshold
        if the number of attempts is greater than the refresh attempts, will give up to refresh
        if less than threshold requires another refresh, it means the proxy pool is not working well, we will give up to refresh
        """
        for url in self.verify_urls:
            if ((valid_proxy_ratio := self.proxies[url].valid_ratio) >= self.refresh_threshold) or not self.refresh_enabled[url]:
                continue
            prefix = f'URL {url} Proxies '
            refresh_time = datetime.now()
            if refresh_time - self.refresh_time[url] < timedelta(seconds=self.refresh_interval):
                Logger.alert1(f"{prefix}refresh re-called too soon, will not refresh anymore")
                self.refresh_enabled[url] = False
                continue
            self.refresh_proxies(url , detail_level = 'none')
            self.refresh_attempt[url] += 1
            if self.proxies[url]:
                Logger.success(f"{prefix}valid ratio drop to {valid_proxy_ratio:.2f}, refresh to {len(self.proxies[url])} proxies")
            if self.refresh_attempt[url] >= self.refresh_max_attempts:
                self.refresh_enabled[url] = False
                Logger.alert1(f"{prefix}refresh count reached max attempts {self.refresh_max_attempts}, will not refresh anymore")
            elif self.is_invalid_url(url):
                self.refresh_enabled[url] = False
                Logger.alert1(f"{prefix}refresh failed with no new proxies, will not refresh anymore")
                
    def execute(self , url_func_tuples : Iterable[ProxyCallerInput] , * , 
                fallback_to_raw_ip: bool = False,
                max_workers : int = 3 , grouping_num : int = 100 , **kwargs) -> list[bool | Exception]:
        callers = ProxyCallerList.from_inputs(url_func_tuples , self)
        groups = callers.partition(grouping_num)
        for group in groups:
            for _ in range(10):
                group.execute(max_workers=max_workers , **kwargs)
                self.refresh()
                if group.is_unable_to_proceed():
                    break
            else:
                Logger.alert2(f"Proxy pool is refreshed too many times, but still able to proceed, WHY?")
            if not group.is_unable_to_proceed() or callers.is_unable_to_proceed():
                break
        if fallback_to_raw_ip and not callers.all_finished:
            callers.fallback()
        return callers.results()

    @classmethod
    def test(cls , * , refresh_interval: int = 1 , refresh_max_attempts: int = 1 , refresh_threshold: float = 0.1):
        return cls('test' , refresh_interval=refresh_interval , refresh_max_attempts=refresh_max_attempts , refresh_threshold=refresh_threshold , invalid_threshold=1 , max_concurrent=2)

    