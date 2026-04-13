"""Run user callables through a ``AdaptiveProxyPool`` with fallback when proxies are missing."""
from __future__ import annotations
import numpy as np
from typing import Callable , Any , Iterable , Union, Iterator

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.proj.log import Logger
from src.proj.util.web import iterate_with_interval_control

ProxyCallerInput = Union[Callable[..., bool | Exception] , tuple[str, Callable[..., bool | Exception]] , tuple[Callable[..., bool | Exception], str] , 'ProxyCaller']

class ProxyDepletionException(Exception):
    """Exception raised when the proxy pool is depleted"""
    pass

class ProxyCaller:
    """A function that can be used to execute a function with a proxy"""
    def __init__(self, func: Callable[..., bool | Exception] , url: str = 'www.example.com' , * , pool = None , title = ''):
        self.func = func
        self.url = url
        self.set_pool(pool)
        self.finished = False
        self.result = False
        self.banned = False
        self.title = title

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.title})"

    def __call__(self, *args: Any, **kwargs: Any) -> bool | Exception:
        if self.url in self.pool.proxies:
            return self.proxied(*args, **kwargs)
        else:
            return self.fallback(*args, **kwargs)

    def set_pool(self , pool = None) -> ProxyCaller:
        """Attach a proxy pool; no-op if one is already set and ``pool`` is None."""
        if pool is None and hasattr(self, 'pool'):
            return self
        from src.proj.util.proxy.api import ProxyAPI
        if pool is None:
            pool = ProxyAPI.get_proxy_pool(self.url)
        self.pool = pool
        return self

    def set_title(self , title: Any) -> ProxyCaller:
        """Set a human-readable label for logging and status display."""
        self.title = str(title)
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

    def fallback(self , *args: Any, **kwargs: Any) -> bool | Exception:
        """Fallback to raw ip"""
        self.result = self.func(None , *args, **kwargs)
        self.finished = True
        return self.result

    @classmethod
    def from_input(cls , input: ProxyCallerInput, pool = None) -> ProxyCaller:
        """
        Construct a ProxyCaller from a flexible input: a bare callable, a (url, func) tuple,
        or an existing ProxyCaller. ``pool`` is required when ``input`` is not already a ProxyCaller.
        """
        if isinstance(input , ProxyCaller):
            if pool is not None:
                input.set_pool(pool)
            return input
        else:
            assert pool is not None , "pool is required"
            if isinstance(input , tuple):
                assert len(input) == 2 , f"input should be a tuple of length 2, contains url and func, but got {input}"
                if isinstance(input[0], str) and isinstance(input[1], Callable):
                    return cls(input[1] , input[0] , pool = pool)
                elif isinstance(input[0], Callable) and isinstance(input[1], str):
                    return cls(input[0] , input[1] , pool = pool)
                else:
                    raise ValueError(f"Invalid input: {input}")
            else:
                assert isinstance(input , Callable) , f"single input should be a callable, but got {input}"
                return cls(input , pool = pool)

class ProxyCallerList:
    """A list of proxy callers"""
    fallback_interval = 1.0
    
    def __init__(self, callers: list[ProxyCaller] | dict[str, ProxyCaller] , * , pool = None):
        if isinstance(callers , dict):
            self.callers = [caller.set_title(title) for title, caller in callers.items()]
        else:
            self.callers = callers
        self.set_pool(pool)

    def __len__(self) -> int:
        return len(self.callers)

    def __bool__(self) -> bool:
        return bool(self.callers)

    def __iter__(self) -> Iterator[ProxyCaller]:
        return iter(self.callers)

    def set_pool(self , pool = None) -> ProxyCallerList:
        """Attach a proxy pool to this list and propagate it to all caller members."""
        if pool is None and hasattr(self, 'pool'):
            return self
        from src.proj.util.proxy.api import ProxyAPI
        if pool is None:
            pool = ProxyAPI.get_proxy_pool(set([caller.url for caller in self.callers]))
        self.pool = pool
        [caller.set_pool(pool) for caller in self.callers]
        return self

    @property
    def all_finished(self) -> bool:
        """True when every caller has completed (either successfully or via fallback)."""
        return len(self) == 0 or all(caller.finished for caller in self.callers)

    def is_unable_to_proceed(self) -> bool:
        """Whether the proxy pool is unable to proceed"""
        self.check_shutdown()
        return len(self) == 0 or all(caller.finished or caller.banned for caller in self.callers)
    
    def unfinished_callers(self) -> list[ProxyCaller]:
        """Return all callers that have not yet finished."""
        return [caller for caller in self.callers if not caller.finished]

    def print_status(self):
        """Print the status of the proxy pool"""
        Logger.stdout([(caller.url, caller.finished, caller.banned) for caller in self.callers])

    def check_shutdown(self):
        """Ban the url"""
        shutdown_urls = [url for url in self.pool.target_urls if self.pool.proxies[url].shutdown]
        [caller.ban() for caller in self.callers if caller.url in shutdown_urls]

    def results(self) -> list[bool | Exception]:
        """Return the result of each caller (True on success, False on failure, or an Exception)."""
        return [caller.result for caller in self.callers]

    def realigned_callers(self , unfinished = True) -> list[ProxyCaller]:
        """
        Re-interleave callers by URL so that consecutive calls hit different targets.

        Uses a stride-based shuffle (step ≈ sqrt(n)) to spread work across URLs more evenly.
        """
        callers = self.unfinished_callers() if unfinished else self.callers
        step_size = int(np.round(np.sqrt(len(callers))))
        sor = sorted(callers , key = lambda x: x.url)
        new = []
        for i in range(0, step_size):
            new.extend(sor[i::step_size])
        assert len(new) == len(new) , (len(new) , len(self.callers))
        return new

    def fallback(self):
        """Fallback to raw ip , sleep for a while to avoid hitting the rate limit"""
        unfinished_callers = iterate_with_interval_control([caller for caller in self.callers if not caller.finished] , interval = self.fallback_interval)
        for caller in unfinished_callers:
            caller.fallback()

    @classmethod
    def from_inputs(cls , inputs : Iterable[ProxyCallerInput] | dict[str, ProxyCallerInput] , pool = None) -> ProxyCallerList:
        """Create a ProxyCallerList from inputs of Iterable[func: Callable[..., bool]] | Iterable[tuple[url: str, func: Callable[..., bool]]]"""
        if isinstance(inputs , ProxyCallerList):
            return inputs.set_pool(pool)
        assert pool is not None , "pool is required"
        if isinstance(inputs , dict):
            callers = [ProxyCaller.from_input(input , pool).set_title(title) for title, input in inputs.items()]
        else:
            callers = [ProxyCaller.from_input(func , pool) for func in inputs]
        self = cls(callers , pool = pool)
        return self

    def partition(self , grouping_num : int = 100) -> list[ProxyCallerList]:
        """Partition the callers into groups"""
        if not self.callers:
            return []
        callers = self.realigned_callers(unfinished = True)
        num_groups = (len(callers) / grouping_num).__ceil__()
        max_callers_per_group = (len(callers) / num_groups).__ceil__()
        groups = [ProxyCallerList(callers[i * max_callers_per_group:(i + 1) * max_callers_per_group] , pool = self.pool) for i in range(num_groups)]
        return groups

    def execute(self , * , max_workers : int = 10):
        """Execute the unfinished callers with a thread pool"""
        unfinished_callers = [caller for caller in self.callers if not caller.finished]
        if not unfinished_callers:
            return
        if max_workers == 1 or len(unfinished_callers) == 1:
            for caller in unfinished_callers:
                caller()
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(unfinished_callers))) as pool:
                fut_map = {pool.submit(caller): caller for caller in unfinished_callers}
                for fut in as_completed(fut_map):
                    fut.result()

    def execute_with_partition(
        self , * , fallback_to_raw_ip: bool = False, max_workers : int = 10 , **kwargs) -> list[bool | Exception]:
        """
        Execute all callers in multiple rounds with adaptive partitioning.

        Partitions callers into sqrt-sized groups, executes each group, triggers an adaptive
        proxy-pool refresh between groups, and repeats for up to 10 rounds until all are done.
        Falls back to direct (no-proxy) calls for any remaining unfinished callers when
        ``fallback_to_raw_ip=True``.
        """
        grouping_num = max(int(np.round(np.sqrt(len(self.callers)))) , 2 * max_workers)
        for i_iter in range(10):
            groups = self.partition(grouping_num)
            Logger.stdout(f"Execute with partitioning in Round {i_iter}, partition into {len(groups)} groups" , indent = 1 , vb_level = 'max')
            for i_group, group in enumerate(groups):
                group.execute(max_workers=max_workers , **kwargs)
                Logger.stdout(f"Finished executing {len(group.callers)} callers for group {i_group}," ,  
                              f"{len(group.unfinished_callers())} unfinished" , indent = 2 , vb_level = 'max')
                self.pool.adaptive_refresh()
                Logger.stdout(f"Try Refresh Proxy Pool for group {i_group} in round {i_iter} finished" , indent = 2 , vb_level = 'max')
            if self.is_unable_to_proceed():
                break
        else:
            Logger.alert2(f"After 10 Rounds, the proxy pool is still able to proceed, WHY?")
            Logger.alert1(self.unfinished_callers())
        if fallback_to_raw_ip and not self.all_finished:
            Logger.alert1(f"Fallback to raw ip for {len(self.unfinished_callers())} callers" , indent = 1 , vb_level = 'max')
            self.fallback()
        return self.results()

    def execute_with_partition_old(
        self , * , fallback_to_raw_ip: bool = False, max_workers : int = 10 , grouping_num : int = 100 , **kwargs) -> list[bool | Exception]:
        """Legacy partitioned execution; kept for reference. Prefer :meth:`execute_with_partition`."""
        groups = self.partition(grouping_num)
        for group in groups:
            for i in range(10):
                group.execute(max_workers=max_workers , **kwargs)
                self.pool.adaptive_refresh()
                if group.is_unable_to_proceed():
                    break
            else:
                Logger.alert2(f"Proxy pool is refreshed too many times, but still able to proceed, WHY?")
            if not group.is_unable_to_proceed() or self.is_unable_to_proceed():
                break
        if fallback_to_raw_ip and not self.all_finished:
            self.fallback()
        return self.results()
