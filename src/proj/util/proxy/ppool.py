"""Adaptive proxy pool: fetch, verify, refresh, and serve URLs for crawlers."""
from __future__ import annotations
import random
import threading
import time
from datetime import datetime , timedelta
from typing import Iterable , Literal

from src.proj.log import Logger
from .core import Proxy , ProxySet , ProxyStats , ProxyStatsSet
from .verifier import ProxyVerifier
from .finder import FreeProxyFinder as ProxyFinder
from .cache import ProxyCache
from .caller import ProxyCallerList , ProxyCallerInput

def get_test_proxies():
    """Test fake proxies"""
    return ProxySet(["http://1.2.3.4:8080", "http://5.6.7.8:3128", "http://9.10.11.12:9999", "http://4.4.4.4:8080", "http://5.5.5.5:3129"])

def get_working_proxies(
    target_url: str , * , min_count: int = 5, max_round: int = 3 , timeout: float = 10.0, workers: int = 50 , 
    dummy: bool = False, go_with_cached_proxies: bool = False, detail_level: Literal['all','none','simple'] = 'all') -> ProxySet:
    """
    return the list of available proxy URLs; with in-process short-term cache to avoid hitting the free proxy site for each failed task.
    if force_refresh is True, ignore the expired cache.
    """
    if target_url == 'test':
        return get_test_proxies()
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


class ProxyStatsSetURL(ProxyStatsSet):
    _instances : dict[str, ProxyStatsSetURL] = {}
    def __new__(cls, *args, **kwargs):
        url = str(kwargs['url'])
        assert url , "not empty url is required"
        if url not in cls._instances:
            cls._instances[url] = super().__new__(cls)
        return cls._instances[url]

    def __init__(
        self , 
        proxies: Iterable[ProxyStats | Proxy | str] | None = None , source: str = 'unknown' , * , 
        url: str , adaptive: bool = False,
        refresh_interval: int = 180 , refresh_max_attempts: int = 10 , refresh_threshold: float = 0.2,
        go_with_cached_proxies: bool = False,
        **kwargs
    ):
        if getattr(self, '_inited', False):
            assert url == self.url , f"url is already set to {self.url} , and cannot be changed to {url}"
            if proxies is not None:
                self.extend(proxies)
            return

        super().__init__(proxies, source)
        self.set_url(url)
        self.adaptive = adaptive
        self.init_refresh_stats(refresh_interval=refresh_interval, refresh_max_attempts=refresh_max_attempts, refresh_threshold=refresh_threshold)
        self.initial_refresh(go_with_cached_proxies)
        self._inited = True

    def __bool__(self) -> bool:
        return not self.shutdown

    def init_refresh_stats(self , refresh_interval: int = 180 , refresh_max_attempts: int = 10 , refresh_threshold: float = 0.2):
        self.refresh_enabled = self.adaptive
        self.refresh_time = datetime.now()
        self.refresh_attempt = 0
        self.refresh_interval = refresh_interval
        self.refresh_max_attempts = refresh_max_attempts
        self.refresh_threshold = refresh_threshold

    def set_url(self, url: str):
        assert not hasattr(self, '_url') , "url is already set"
        self._url = url
        return self

    @property
    def url(self) -> str:
        return self._url

    @property
    def shutdown(self) -> bool:
        """Whether the proxy pool is shutdown"""
        return self.valid_count == 0 and not (self.adaptive and self.refresh_enabled)

    def check_validity(self) -> bool:
        if self.shutdown:
            Logger.only_once(f"ProxySet for URL {self.url} shutdown , all proxies are invalid and refresh is disabled" , object = self , mark = 'shutdown_info' , printer = Logger.alert2)
            return False
        return self.valid_count > 0

    def refresh_proxies(self , go_with_cached_proxies: bool = False , detail_level: Literal['all','none','simple'] = 'all'):
        """Get the working proxies from the pool's proxy API"""
        old_proxies = [proxy for proxy in self.proxies if proxy.invalid]
        new_proxies = [ProxyStats(proxy) for proxy in get_working_proxies(self.url, go_with_cached_proxies = go_with_cached_proxies , detail_level = detail_level)]
        self.proxies = list(set(old_proxies + new_proxies))
        self.refresh_time = datetime.now()

    def print_status(self):
        """Print the status of the proxy pool"""
        Logger.stdout(f"URL {self.url} has {len(self.proxies)} proxies, {self.valid_ratio:.2%} valid")

    def initial_refresh(self , go_with_cached_proxies: bool = False):
        """Refresh the proxy set with new proxies"""
        self.refresh_proxies(go_with_cached_proxies=go_with_cached_proxies , detail_level = 'simple')

    def adaptive_refresh(self):
        """
        Refresh the proxy set with new proxies under certain circumstances:
        - will only refresh if valid proxy ratio is less than the refresh threshold
        - if the number of attempts is greater than the refresh attempts, will give up to refresh
        - if less than threshold requires another refresh, it means the proxy pool is not working well, we will give up to refresh
        """
        valid_ratio = self.valid_ratio
        if (valid_ratio >= self.refresh_threshold) or not self.refresh_enabled:
            return
        prefix = f'URL {self.url} Proxies '
        refresh_time = datetime.now()
        if refresh_time - self.refresh_time < timedelta(seconds=self.refresh_interval):
            Logger.alert1(f"{prefix}refresh re-called too soon, will not refresh anymore")
            self.refresh_enabled = False
            return
        self.refresh_proxies(detail_level = 'none')
        self.refresh_attempt += 1
        if self.proxies:
            Logger.success(f"{prefix}valid ratio drop to {valid_ratio:.2f}, refresh to {len(self.proxies)} proxies (attempt {self.refresh_attempt}/{self.refresh_max_attempts})")
        if self.refresh_attempt >= self.refresh_max_attempts:
            self.refresh_enabled = False
            Logger.alert1(f"{prefix}refresh count reached max attempts {self.refresh_max_attempts}, will not refresh anymore")
        elif self.valid_ratio == 0:
            self.refresh_enabled = False
            Logger.alert1(f"{prefix}refresh failed with no new proxies, will not refresh anymore")

class ProxyPool:
    """Proxy pool, can be used to acquire and release proxies in a thread-safe manner"""
    def __init__(self, target_urls: list[str] | str , * , go_with_cached_proxies: bool = False):
        self.initiate(target_urls, go_with_cached_proxies = go_with_cached_proxies , adaptive=False)

    def initiate(self , target_urls: list[str] | str , **kwargs):
        if target_urls == 'test':
            Logger.alert1("Using test mode and pseudo proxies")
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.target_urls = target_urls if isinstance(target_urls , list) else [target_urls]
        self.proxies = {url: ProxyStatsSetURL(url=url , **kwargs) for url in self.target_urls}
        
    def __len__(self) -> int:
        return len(self.target_urls)

    def __bool__(self) -> bool:
        return not self.shutdown

    @property
    def num_proxies(self) -> dict[str, int]:
        return {url: len(self.proxies[url]) for url in self.target_urls}

    @property
    def shutdown(self) -> bool:
        """Whether the proxy pool is shutdown"""
        return all(proxy_set.shutdown for proxy_set in self.proxies.values())

    def print_status(self):
        """Print the status of the proxy pool"""
        for proxy_set in self.proxies.values():
            proxy_set.print_status()
        
    def acquire(self , url: str) -> ProxyStats | None:
        """Acquire a proxy, if no available proxy but some proxies are not invalid, wait until a proxy is available, otherwise return None"""
        with self.condition:
            while True:
                # condition 1: if the url proxy set is not valid, return None
                if not self.proxies[url].check_validity():
                    return None

                # condition 2: if there are available proxies, randomly select one and return it
                if proxy := self.proxies[url].pick_one():
                    return proxy

                # condition 3: if there are no available proxies, wait until a proxy is available
                Logger.footnote(f"URL [{url}] is waiting for a proxy")
                self.condition.wait()

    def release(self, proxy: ProxyStats, success: bool) -> None:
        """Release a proxy, and update the state of this usage."""
        with self.condition:
            proxy.release(success)
            # notify all waiting threads that the proxy state has changed
            self.condition.notify_all()

    def execute(self , caller_inputs : Iterable[ProxyCallerInput] , * , max_workers : int = 3 , grouping_num : int = 100 , **kwargs) -> list[bool | Exception]:
        callers = ProxyCallerList.from_inputs(caller_inputs , self)
        groups = callers.partition(grouping_num)
        for group in groups:
            group.execute(max_workers=max_workers , **kwargs)
            if group.is_unable_to_proceed():
                break
        return callers.results()

    @classmethod
    def test(cls):
        """return a test proxy pool"""
        return cls('test')

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
    """
    def __init__(
        self, target_urls: list[str] | str , * , 
        go_with_cached_proxies: bool = False,
        refresh_interval: int = 180 ,
        refresh_max_attempts: int = 10 ,
        refresh_threshold: float = 0.2 ,
    ):
        self.initiate(target_urls, adaptive=True , go_with_cached_proxies = go_with_cached_proxies , 
                      refresh_interval=refresh_interval , refresh_max_attempts=refresh_max_attempts , refresh_threshold=refresh_threshold)

    def adaptive_refresh(self):
        """Refresh the proxy pool with new proxies"""
        for proxy_set in self.proxies.values():
            proxy_set.adaptive_refresh()
                
    def execute(self , caller_inputs : Iterable[ProxyCallerInput] , * , 
                fallback_to_raw_ip: bool = False,
                max_workers : int = 3 , grouping_num : int = 100 , **kwargs) -> list[bool | Exception]:
        callers = ProxyCallerList.from_inputs(caller_inputs , self)
        groups = callers.partition(grouping_num)
        for group in groups:
            for _ in range(10):
                group.execute(max_workers=max_workers , **kwargs)
                self.adaptive_refresh()
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
        return cls('test' , refresh_interval=refresh_interval , refresh_max_attempts=refresh_max_attempts , refresh_threshold=refresh_threshold)

    