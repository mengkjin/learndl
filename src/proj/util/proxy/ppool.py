import random
import threading
import time
import numpy as np
from datetime import datetime , timedelta
from typing import Callable , Any , Iterable , Literal

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.proj.log import Logger
from .verifier import ProxyVerifier

MAX_WORKERS : int = 3 # max workers for the proxy pool for a single url, cannot set to large as it will cause the proxy pool to be overwhelmed

def test_proxies():
    """Test fake proxies"""
    return [
        "1.2.3.4:8080",
        "5.6.7.8:3128",
        "9.10.11.12:9999",
        "4.4.4.4:8080",
        "5.5.5.5:3129",
    ]

class ProxyWithStats:
    """A stated proxy, can be used to track the state of a proxy"""
    invalid_threshold : int = 3
    max_concurrent : int = 2

    _instances : dict[str, 'ProxyWithStats'] = {}

    def __new__(cls, addr: str):
        if addr not in cls._instances:
            cls._instances[addr] = super().__new__(cls)
            cls._instances[addr].addr = addr
        return cls._instances[addr]

    def __repr__(self) -> str:
        return f"StatedProxy(addr={self.addr})"

    @property
    def addr(self) -> str:
        return self._addr

    @addr.setter
    def addr(self, value: str):
        self._addr = value

    @property
    def stats(self) -> dict[Literal['occupied', 'error', 'success'], int]:
        if not hasattr(self, '_stats'):
            self._stats : dict[Literal['occupied', 'error', 'success'], int] = {
                'occupied': 0,
                'error': 0,
                'success': 0,
            }
        return self._stats

    @classmethod
    def set_class_attrs(cls, invalid_threshold: int | None = None, max_concurrent: int | None = None):
        if invalid_threshold is not None:
            cls.invalid_threshold = invalid_threshold
        if max_concurrent is not None:
            cls.max_concurrent = max_concurrent

    @property
    def valid(self) -> bool:
        """Whether the proxy is valid"""
        return self.stats['error'] < self.invalid_threshold

    @property
    def invalid(self) -> bool:
        """Whether the proxy is invalid (error count >= invalid threshold)"""
        return self.stats['error'] >= self.invalid_threshold

    @property
    def available(self) -> bool:
        """Whether the proxy is available (valid and occupied < max concurrent)"""
        return self.valid and self.stats['occupied'] < self.max_concurrent

    @property
    def total_count(self) -> int:
        """The total count of the proxy (success + error)"""
        return self.stats['success'] + self.stats['error']

    def acquire(self):
        """Acquired, increment the occupied count"""
        self.stats['occupied'] += 1
        return self

    def release(self, success: bool):
        """Released, decrement the occupied count, and update the success or error count"""
        self.stats['occupied'] -= 1
        if success:
            self.stats['success'] += 1
        else:
            self.stats['error'] += 1

class ProxyCaller:
    """A function that can be used to execute a function with a proxy"""
    def __init__(self, pool: 'ProxyPoolMultiURL | ProxyPool', func: Callable[..., bool] , url: str = ''):
        self.pool = pool
        self.func = func
        self.url = url
        self.finished = False
        self.result = None
        self.banned = False

    def __call__(self, *args: Any, **kwargs: Any) -> bool | None:
        return self.proxied(*args, **kwargs)

    def ban(self):
        """Ban the caller"""
        self.banned = True

    def proxied(self , *args: Any, **kwargs: Any) -> bool | None:
        """Call the function with a proxy"""
        proxy = self.pool.acquire(self.url)
        if proxy is None:
            return None
        self.result = self.func(proxy.addr , *args, **kwargs)
        self.finished = True
        self.pool.release(proxy, self.result)
        return self.result

    def fallback(self , *args: Any, **kwargs: Any) -> bool:
        """Fallback to raw ip"""
        self.result = self.func(None , *args, **kwargs)
        self.finished = True
        return self.result

class ProxyCallerList:
    """A list of proxy callers"""
    def __init__(self, callers: list['ProxyCaller'] , pool: 'ProxyPoolMultiURL | ProxyPool'):
        self.callers = callers
        self.pool = pool

    def __len__(self) -> int:
        return len(self.callers)

    def __bool__(self) -> bool:
        return bool(self.callers)

    @property
    def all_finished(self) -> bool:
        return len(self) == 0 or all(caller.finished for caller in self.callers)

    def is_unable_to_proceed(self) -> bool:
        """Whether the proxy pool is unable to proceed"""
        self.check_shutdown()
        return len(self) == 0 or all(caller.finished or caller.banned for caller in self.callers)

    def check_shutdown(self):
        """Ban the url"""
        if isinstance(self.pool , ProxyPool) and self.pool.shutdown:
            [caller.ban() for caller in self.callers]
        elif isinstance(self.pool , ProxyPoolMultiURL):
            shutdown_urls = [url for url in self.pool.verify_urls if self.pool.url_shutdown(url)]
            [caller.ban() for caller in self.callers if caller.url in shutdown_urls]

    def results(self) -> list[bool | None]:
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

    def execute(self , * , max_workers : int = MAX_WORKERS):
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
        """Fallback to raw ip"""
        for caller in self.callers:
            if not caller.finished:
                caller.fallback()

    @classmethod
    def from_inputs(
        cls , pool: 'ProxyPoolMultiURL | ProxyPool', 
        inputs : Iterable[Callable[..., bool]] | Iterable[tuple[str, Callable[..., bool]]]
    ) -> 'ProxyCallerList':
        """Create a ProxyCallerList from inputs of Iterable[func: Callable[..., bool]] | Iterable[tuple[url: str, func: Callable[..., bool]]]"""
        callers : list[ProxyCaller] = []
        for func in inputs:
            if isinstance(func , tuple):
                callers.append(ProxyCaller(pool , func[1] , func[0]))
            else:
                callers.append(ProxyCaller(pool , func))
        self = cls(callers , pool)
        return self

    def partition(self , grouping_num : int = 100) -> list['ProxyCallerList']:
        """Partition the callers into groups"""
        if not self.callers:
            return []
        callers = self.realigned_callers()
        num_groups = np.ceil(len(callers) / grouping_num)
        max_callers_per_group = np.ceil(len(callers) / num_groups)
        groups = []
        for i in range(num_groups):
            groups.append(ProxyCallerList(callers[i * max_callers_per_group:(i + 1) * max_callers_per_group] , self.pool))
        return groups

class ProxyPool:
    """Proxy pool, can be used to acquire and release proxies in a thread-safe manner"""
    def __init__(self, proxies: dict[str , list[str]] | list[str] , * , invalid_threshold: int | None = None, max_concurrent: int | None = None):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        ProxyWithStats.set_class_attrs(invalid_threshold = invalid_threshold, max_concurrent = max_concurrent)
        self.proxies: list[ProxyWithStats] = [ProxyWithStats(proxy) for proxy in proxies]
        self.invalid_printed = False

    def __len__(self) -> int:
        return len(self.proxies)

    def __bool__(self) -> bool:
        return not self.shutdown

    @property
    def shutdown(self) -> bool:
        """Whether the proxy pool is shutdown"""
        return self.valid_proxy_ratio() == 0.0

    def url_shutdown(self , *args) -> bool:
        """Whether the url is shutdown"""
        return self.shutdown

    def valid_proxy_ratio(self) -> float:
        """The ratio of invalid proxies"""
        return 0.0 if not self.proxies else sum(proxy.valid for proxy in self.proxies) / len(self.proxies)

    def pick_a_proxy(self) -> ProxyWithStats | None:
        """Get available proxies"""
        available_proxies = [proxy for proxy in self.proxies if proxy.available]
        if available_proxies:
            proxy = random.choice(available_proxies)
            proxy.acquire()
            return proxy
        return None

    def acquire(self , *args) -> ProxyWithStats | None:
        """Acquire a proxy, if no available proxy but some proxies are not invalid, wait until a proxy is available, otherwise return None"""
        with self.condition:
            while True:
                # condition 1: if all proxies are invalid, return None
                if self.shutdown:
                    if not self.invalid_printed:
                        Logger.alert2(f"All proxies are invalid, return None")
                        self.invalid_printed = True
                    return None

                # condition 2: if there are available proxies, randomly select one and return it
                if proxy := self.pick_a_proxy():
                    return proxy

                # condition 3: if there are no available proxies, wait until a proxy is available
                self.condition.wait()

    def release(self, proxy: ProxyWithStats, success: bool) -> None:
        """Release a proxy, and update the state of this usage."""
        with self.condition:
            proxy.release(success)
            # notify all waiting threads that the proxy state has changed
            self.condition.notify_all()

    def execute(self , funcs : Iterable[Callable[..., bool]] , * , max_workers : int = MAX_WORKERS , grouping_num : int = 100 , **kwargs) -> list[bool | None]:
        callers = ProxyCallerList.from_inputs(self , funcs)
        groups = callers.partition(grouping_num)
        for group in groups:
            group.execute(max_workers=max_workers , **kwargs)
            if group.is_unable_to_proceed():
                break
        return callers.results()

    @classmethod
    def test(cls):
        """return a test proxy pool"""
        return cls(test_proxies() , invalid_threshold=1 , max_concurrent=2)

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
        workers = [lambda proxy, i=i: test_worker(proxy, worker_id=i) for i in range(4 * len(self))]
        return self.execute(workers , max_workers=max_workers)

class ProxyPoolMultiURL:
    """Proxy pool, can be used to acquire and release proxies in a thread-safe manner"""
    def __init__(self, target_url: list[str] | str | Literal['test'], * , 
                 proxies : dict[Any , list[str]] | None = None ,
                 go_with_cached_proxies: bool = False,
                 invalid_threshold: int | None = None, max_concurrent: int | None = None):
        if target_url == 'test':
            Logger.alert1("Using test mode and pseudo proxies")
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.verify_urls = target_url if isinstance(target_url , list) else [target_url]
        ProxyWithStats.set_class_attrs(invalid_threshold = invalid_threshold, max_concurrent = max_concurrent)
        self.proxies : dict[str , list[ProxyWithStats]] = {url: [] for url in self.verify_urls}
        self.refresh_time = {url: datetime.now() for url in self.verify_urls}
        self.invalid_printed = {url: False for url in self.verify_urls}
        self.refresh_proxies(url = 'all' , go_with_cached_proxies=go_with_cached_proxies , verbose = 1)
        
    def refresh_proxies(self , url : str | Literal['all' , 'test'] , go_with_cached_proxies: bool = False , verbose: int = 3):
        """Get the working proxies from the pool's proxy API"""
        if url == 'all':
            urls = self.verify_urls
        else:
            urls = [url]
        with self.condition:
            for url in urls:
                enabled_proxies = [proxy for proxy in self.proxies.get(url, []) if not proxy.invalid]
                new_proxies = self.get_working_proxies(url, go_with_cached_proxies , verbose = verbose)
                new_proxies = [ProxyWithStats(proxy) for proxy in new_proxies]
                self.proxies[url] = enabled_proxies + new_proxies
                self.refresh_time[url] = datetime.now()

    @classmethod
    def get_working_proxies(cls , url : str , go_with_cached_proxies: bool = False , verbose: int = 3) -> list[str]:
        """Get the working proxies from the pool's proxy API"""
        if url == 'test':
            return test_proxies()
        else:
            return ProxyVerifier.get_working_proxies(url, go_with_cached_proxies=go_with_cached_proxies , verbose=verbose)

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
        return self.valid_proxy_ratio(url) == 0

    def valid_proxy_ratio(self , url: str) -> float:
        """The ratio of invalid proxies"""
        return 0 if not self.proxies[url] else sum(proxy.valid for proxy in self.proxies[url]) / len(self.proxies[url])

    def pick_a_proxy(self , url: str) -> ProxyWithStats | None:
        """Get available proxies"""
        available_proxies = [proxy for proxy in self.proxies[url] if proxy.available]
        if available_proxies:
            proxy = random.choice(available_proxies)
            proxy.acquire()
            return proxy
        return None

    def acquire(self , url: str) -> ProxyWithStats | None:
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
                if proxy := self.pick_a_proxy(url):
                    return proxy

                # condition 3: if there are no available proxies, wait until a proxy is available
                self.condition.wait()

    def release(self, proxy: ProxyWithStats, success: bool) -> None:
        """Release a proxy, and update the state of this usage."""
        with self.condition:
            proxy.release(success)
            # notify all waiting threads that the proxy state has changed
            self.condition.notify_all()

    def execute(self , url_func_tuples : Iterable[tuple[str, Callable[..., bool]]] , * , max_workers : int = MAX_WORKERS , grouping_num : int = 100 , **kwargs) -> list[bool | None]:
        callers = ProxyCallerList.from_inputs(self , url_func_tuples)
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

class ProxyPoolAutoRefresh(ProxyPoolMultiURL):
    """
    Proxy pool that can be automatically replenished with new proxies
    init args:
        target_url: the url to verify the proxies, i.e. the target url to access
        refresh_interval: the minimum interval to refresh the proxies , if all proxies down before the interval, will exit the pool
        refresh_attempts: the maximum attempts to refresh the proxies
        refresh_threshold: the ratio of initial proxies that are remained enabled to trigger a refresh
        invalid_threshold: the threshold to mark the proxies as invalid
        max_concurrent: the max concurrent proxies to use
    """
    def __init__(
        self, target_url: list[str] | str | Literal['test'] , * , proxies : dict[str , list[str]] | None = None ,
        go_with_cached_proxies: bool = False,
        refresh_interval: int = 180 ,
        refresh_max_attempts: int = 10 ,
        refresh_threshold: float = 0.2 ,
        invalid_threshold: int = 3 , max_concurrent: int = 2
    ):
        super().__init__(
            target_url , 
            proxies=proxies , 
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
            if ((valid_proxy_ratio := self.valid_proxy_ratio(url)) >= self.refresh_threshold) or not self.refresh_enabled[url]:
                continue
            prefix = f'URL {url} Proxies '
            refresh_time = datetime.now()
            if refresh_time - self.refresh_time[url] < timedelta(seconds=self.refresh_interval):
                Logger.alert1(f"{prefix}refresh re-called too soon, will not refresh anymore")
                self.refresh_enabled[url] = False
                continue
            self.refresh_proxies(url , verbose = 0)
            self.refresh_attempt[url] += 1
            if self.proxies[url]:
                Logger.success(f"{prefix}valid ratio drop to {valid_proxy_ratio:.2f}, refresh to {len(self.proxies[url])} proxies")
            if self.refresh_attempt[url] >= self.refresh_max_attempts:
                self.refresh_enabled[url] = False
                Logger.alert1(f"{prefix}refresh count reached max attempts {self.refresh_max_attempts}, will not refresh anymore")
            elif self.is_invalid_url(url):
                self.refresh_enabled[url] = False
                Logger.alert1(f"{prefix}refresh failed with no new proxies, will not refresh anymore")
                
    def execute(self , url_func_tuples : Iterable[tuple[str, Callable[..., bool]]] , * , 
                fallback_to_raw_ip: bool = False,
                max_workers : int = MAX_WORKERS , grouping_num : int = 100 , **kwargs) -> list[bool | None]:
        callers = ProxyCallerList.from_inputs(self , url_func_tuples)
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

    