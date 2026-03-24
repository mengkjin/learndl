import random
import threading
import time
from datetime import datetime , timedelta
from typing import Callable , Any , Iterable , Literal

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.proj.log import Logger
from .verifier import ProxyVerifier

MAX_WORKERS : int = 3 # max workers for the proxy pool for a single url, cannot set to large as it will cause the proxy pool to be overwhelmed

class StatedProxy:
    """A stated proxy, can be used to track the state of a proxy"""
    invalid_threshold : int = 3
    max_concurrent : int = 2

    _instances : dict[str, 'StatedProxy'] = {}

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

class ProxyPool:
    """Proxy pool, can be used to acquire and release proxies in a thread-safe manner"""
    def __init__(self, proxies: dict[str , list[str]] | list[str] , * , invalid_threshold: int | None = None, max_concurrent: int | None = None):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        StatedProxy.set_class_attrs(invalid_threshold = invalid_threshold, max_concurrent = max_concurrent)
        self.proxies: list[StatedProxy] = [StatedProxy(proxy) for proxy in proxies]
        self.invalid_printed = False

    def __len__(self) -> int:
        return len(self.proxies)

    def __bool__(self) -> bool:
        return not self._is_all_invalid()

    @property
    def shutdown(self) -> bool:
        """Whether the proxy pool is shutdown"""
        return self._is_all_invalid()

    def _is_all_invalid(self) -> bool:
        """Whether all proxies are invalid"""
        return all(proxy.invalid for proxy in self.proxies)

    def _valid_proxy_ratio(self) -> float:
        """The ratio of invalid proxies"""
        return sum(proxy.valid for proxy in self.proxies) / len(self.proxies)

    def _pick_a_proxy(self) -> StatedProxy | None:
        """Get available proxies"""
        available_proxies = [proxy for proxy in self.proxies if proxy.available]
        if available_proxies:
            proxy = random.choice(available_proxies)
            proxy.acquire()
            return proxy
        return None

    def acquire(self) -> StatedProxy | None:
        """Acquire a proxy, if no available proxy but some proxies are not invalid, wait until a proxy is available, otherwise return None"""
        with self.condition:
            while True:
                # condition 1: if all proxies are invalid, return None
                if self._is_all_invalid():
                    if not self.invalid_printed:
                        Logger.alert2(f"All proxies are invalid, return None")
                        self.invalid_printed = True
                    return None

                # condition 2: if there are available proxies, randomly select one and return it
                if proxy := self._pick_a_proxy():
                    return proxy

                # condition 3: if there are no available proxies, wait until a proxy is available
                self.condition.wait()

    def release(self, proxy: StatedProxy, success: bool) -> None:
        """Release a proxy, and update the state of this usage."""
        with self.condition:
            proxy.release(success)
            # notify all waiting threads that the proxy state has changed
            self.condition.notify_all()

    def join(self , func: Callable[..., bool] , func_name_suffix: str = '') -> Callable[..., bool | None]:
        """
        Join a function to the proxy pool, return a new function
        input function should have the first argument as a proxy, and return a bool value (True if success, False if failed, cannot return None)
        return a new function without the proxy argument as it will be acquired from the proxy pool, and return a bool value
        """
        def wrapper(*args: Any, **kwargs: Any) -> bool | None:
            proxy = self.acquire()
            if proxy is None:
                return None
            ret = func(proxy.addr , *args, **kwargs)
            self.release(proxy, ret)
            return ret
        wrapper.__name__ = f'{func.__name__}{func_name_suffix}'
        return wrapper

    def execute_group(self , calls : Iterable[tuple[int, Callable[..., bool | None]]] , results : list[bool | None] , * , max_workers : int = MAX_WORKERS , **kwargs) -> bool:
        unfinished_calls = [(i, call) for i, call in calls if results[i] is None]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_map = {pool.submit(call , **kwargs): (i,call) for i, call in unfinished_calls}
            for fut in as_completed(fut_map):
                results[fut_map[fut][0]] = fut.result()
        return not any(results[i] is None for i, _ in unfinished_calls)

    def execute(self , funcs : Iterable[Callable[..., bool]] , * , max_workers : int = MAX_WORKERS , grouping_num : int = 100 , **kwargs) -> list[bool | None]:
        calls = [(i , self.join(func , f'_{i}')) for i , func in enumerate(funcs)]
        results : list[bool | None] = [None for _ in range(len(calls))]
        groups : list[list[tuple[int, Callable[..., bool | None]]]] = [[] for _ in range(len(calls) // grouping_num + 1)]
        for i in range(len(calls)):
            groups[i % len(groups)].append(calls[i])
        for group in groups:
            self.execute_group(group , results , max_workers=max_workers , **kwargs)
            if self._is_all_invalid():
                break
        return results

    @classmethod
    def test_proxies(cls):
        """Test fake proxies"""
        return [
            "1.2.3.4:8080",
            "5.6.7.8:3128",
            "9.10.11.12:9999",
            "4.4.4.4:8080",
            "5.5.5.5:3129",
        ]

    @classmethod
    def test(cls):
        """return a test proxy pool"""
        return cls(cls.test_proxies() , invalid_threshold=1 , max_concurrent=2)

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
        StatedProxy.set_class_attrs(invalid_threshold = invalid_threshold, max_concurrent = max_concurrent)
        self.proxies : dict[str , list[StatedProxy]] = {url: [] for url in self.verify_urls}
        self.refresh_time = {url: datetime.now() for url in self.verify_urls}
        self.invalid_printed = {url: False for url in self.verify_urls}
        self.refresh_proxies(url = 'all' , input_proxies=proxies , go_with_cached_proxies=go_with_cached_proxies)
        
    def refresh_proxies(self , url : str | Literal['all' , 'test'] , input_proxies : dict[str , list[str]] | None = None ,
                        go_with_cached_proxies: bool = False):
        """Get the working proxies from the pool's proxy API"""
        input_proxies = input_proxies or {}
        if url == 'all':
            urls = self.verify_urls
        else:
            urls = [url]
        with self.condition:
            for url in urls:
                enabled_proxies = [proxy for proxy in self.proxies.get(url, []) if not proxy.invalid]
                new_proxies = input_proxies[url] if url in input_proxies else self.get_working_proxies(url, go_with_cached_proxies)
                new_proxies = [StatedProxy(proxy) for proxy in new_proxies]
                self.proxies[url] = enabled_proxies + new_proxies
                self.refresh_time[url] = datetime.now()

    @classmethod
    def get_working_proxies(cls , url : str , go_with_cached_proxies: bool = False) -> list[str]:
        """Get the working proxies from the pool's proxy API"""
        if url == 'test':
            return cls.test_proxies()
        else:
            return ProxyVerifier.get_working_proxies(url, go_with_cached_proxies=go_with_cached_proxies)

    def __len__(self) -> int:
        return len(self.verify_urls)

    def __bool__(self) -> bool:
        return not self._is_all_invalid()

    @property
    def num_proxies(self) -> dict[str, int]:
        return {url: len(self.proxies[url]) for url in self.verify_urls}

    @property
    def shutdown(self) -> bool:
        """Whether the proxy pool is shutdown"""
        return self._is_all_invalid()

    def _is_all_invalid(self) -> bool:
        """Whether all proxies are invalid"""
        return all(self.is_invalid_url(url) for url in self.verify_urls)

    def is_invalid_url(self , url: str) -> bool:
        """Whether the url is invalid"""
        return all(proxy.invalid for proxy in self.proxies[url])

    def _valid_proxy_ratio(self , url: str) -> float:
        """The ratio of invalid proxies"""
        return sum(proxy.valid for proxy in self.proxies[url]) / len(self.proxies[url])

    def _pick_a_proxy(self , url: str) -> StatedProxy | None:
        """Get available proxies"""
        available_proxies = [proxy for proxy in self.proxies[url] if proxy.available]
        if available_proxies:
            proxy = random.choice(available_proxies)
            proxy.acquire()
            return proxy
        return None

    def acquire(self , url: str) -> StatedProxy | None:
        """Acquire a proxy, if no available proxy but some proxies are not invalid, wait until a proxy is available, otherwise return None"""
        with self.condition:
            while True:
                # condition 1: if all proxies for the url are invalid, return None
                if self.is_invalid_url(url):
                    if not self.invalid_printed[url]:
                        Logger.alert2(f"All proxies for url {url} are invalid, return None")
                        self.invalid_printed[url] = True
                    return None

                # condition 2: if there are available proxies, randomly select one and return it
                if proxy := self._pick_a_proxy(url):
                    return proxy

                # condition 3: if there are no available proxies, wait until a proxy is available
                self.condition.wait()

    def release(self, proxy: StatedProxy, success: bool) -> None:
        """Release a proxy, and update the state of this usage."""
        with self.condition:
            proxy.release(success)
            # notify all waiting threads that the proxy state has changed
            self.condition.notify_all()

    def join(self , func: Callable[..., bool] , url: str , func_name_suffix: str = '') -> Callable[..., bool | None]:
        """
        Join a function to the proxy pool, return a new function
        input function should have the first argument as a proxy, and return a bool value
        return a new function without the proxy argument as it will be acquired from the proxy pool, and return a bool value
        """
        def wrapper(*args: Any, **kwargs: Any) -> bool | None:
            proxy = self.acquire(url)
            if proxy is None:
                return None
            ret = func(proxy.addr , *args, **kwargs)
            self.release(proxy, ret)
            return ret
        wrapper.__name__ = f'{func.__name__}{func_name_suffix}'
        return wrapper

    def execute_group(self , calls : Iterable[tuple[int, Callable[..., bool | None]]] , results : list[bool | None] , * , max_workers : int = MAX_WORKERS , **kwargs) -> bool:
        unfinished_calls = [(i, call) for i, call in calls if results[i] is None]
        if max_workers == 1:
            for i, call in unfinished_calls:
                results[i] = call(**kwargs)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                fut_map = {pool.submit(call , **kwargs): (i,call) for i, call in unfinished_calls}
                for fut in as_completed(fut_map):
                    results[fut_map[fut][0]] = fut.result()
        return not any(results[i] is None for i, _ in unfinished_calls)

    def execute(self , url_func_tuples : Iterable[tuple[str, Callable[..., bool]]] , * , max_workers : int = MAX_WORKERS , grouping_num : int = 100 , **kwargs) -> list[bool | None]:
        calls = [(i , self.join(func , url , f'_{i}')) for i , (url, func) in enumerate(url_func_tuples)]
        results : list[bool | None] = [None for _ in range(len(calls))]
        groups : list[list[tuple[int, Callable[..., bool | None]]]] = [[] for _ in range(len(calls) // grouping_num + 1)]
        for i in range(len(calls)):
            groups[i % len(groups)].append(calls[i])
        for group in groups:
            self.execute_group(group , results , max_workers=max_workers , **kwargs)
            if self.shutdown:
                return results
        return results

    @classmethod
    def test_proxies(cls):
        """Test fake proxies"""
        return [
            "1.2.3.4:8080",
            "5.6.7.8:3128",
            "9.10.11.12:9999",
            "4.4.4.4:8080",
            "5.5.5.5:3129",
        ]

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

    @property
    def shutdown(self) -> bool:
        """Whether the proxy pool is shutdown"""
        return self._is_all_invalid() and not any(enabled for enabled in self.refresh_enabled.values())

    def refresh(self):
        """
        Refresh the proxy pool with new proxies
        will only refresh if valid proxy ratio is less than the refresh threshold
        if the number of attempts is greater than the refresh attempts, will give up to refresh
        if less than threshold requires another refresh, it means the proxy pool is not working well, we will give up to refresh
        """
        for url in self.verify_urls:
            valid_proxy_ratio = self._valid_proxy_ratio(url)
            if self.refresh_enabled[url] and valid_proxy_ratio < self.refresh_threshold:
                refresh_time = datetime.now()
                if self.refresh_attempt[url] >= self.refresh_max_attempts or refresh_time - self.refresh_time[url] < timedelta(seconds=self.refresh_interval):
                    Logger.alert2(
                        f"Proxy valid ratio {valid_proxy_ratio:.2f} is dropped below threshold {self.refresh_threshold:.2f} ," , 
                        f"but cannot refresh due to too many attempts ({self.refresh_attempt[url]}) or refresh interval not reached ({(refresh_time - self.refresh_time[url]).total_seconds():.1f} seconds)"
                    )
                    self.refresh_enabled[url] = False
                    continue
                self.refresh_proxies(url)
                self.refresh_attempt[url] += 1
                Logger.success(f"Proxy valid ratio {valid_proxy_ratio:.2f} is dropped below threshold {self.refresh_threshold:.2f} , refreshed proxy pool attempt {self.refresh_attempt[url]}")

    def execute(self , url_func_tuples : Iterable[tuple[str, Callable[..., bool]]]  , * , max_workers : int = MAX_WORKERS , grouping_num : int = 100 , **kwargs) -> list[bool | None]:
        calls = [(i , self.join(func , url , f'_{i}')) for i , (url, func) in enumerate(url_func_tuples)]
        results : list[bool | None] = [None for _ in range(len(calls))]
        groups : list[list[tuple[int, Callable[..., bool | None]]]] = [[] for _ in range(len(calls) // grouping_num + 1)]
        for i in range(len(calls)):
            groups[i % len(groups)].append(calls[i])

        for group in groups:
            while True:
                group_executed = self.execute_group(group , results , max_workers=max_workers , **kwargs)
                self.refresh()
                if group_executed:
                    break
                if self.shutdown:
                    return results
        return results

    @classmethod
    def test(cls , * , refresh_interval: int = 1 , refresh_max_attempts: int = 1 , refresh_threshold: float = 0.1):
        return cls('test' , refresh_interval=refresh_interval , refresh_max_attempts=refresh_max_attempts , refresh_threshold=refresh_threshold , invalid_threshold=1 , max_concurrent=2)

    