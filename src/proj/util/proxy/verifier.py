import time
import random
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

from src.proj.log import Logger
from src.proj.util.http import http_session , test_connection

@dataclass
class VerifRecord:
    target_url: str
    proxy_url: str
    status: bool = False
    start_time: float = 0
    end_time: float = 0

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()

    def set_status(self, status: bool):
        self.status = status

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def info(self) -> str:
        return f'{self.target_url} {self.proxy_url} {self.status} {self.duration}'

    def to_dict(self) -> dict:
        return {
            'target_url': self.target_url,
            'proxy_url': self.proxy_url,
            'status': self.status,
            'duration': self.duration,
        }

class VerificationRecords:
    def __init__(self):
        self.records : dict[tuple[str, str], VerifRecord] = {}
        self.lock = threading.Lock()

    def get_record(self, target_url: str, proxy_url: str) -> VerifRecord | bool:
        status = self.get_status(target_url, proxy_url)
        if status is not None:
            return status
        record = VerifRecord(target_url, proxy_url)
        with self.lock:
            self.records[(target_url, proxy_url)] = record
        return record

    def get_status(self, target_url: str, proxy_url: str) -> bool | None:
        with self.lock:
            if (target_url, proxy_url) in self.records:
                return self.records[(target_url, proxy_url)].status
            return None

    def stats(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame([record.to_dict() for record in self.records.values()])
        groups = ['target_url' , 'status']
        return df.groupby(groups).agg({'proxy_url': 'count'}).join(
            df.groupby(groups)['duration'].mean().rename('avg_time')).join(
            df.groupby(groups)['duration'].max().rename('max_time')).join(
            df.groupby(groups)['duration'].quantile(0.9).rename('90%_time'))

class ProxyVerifier:
    """Proxy verifier, can be used to verify single or multiple proxies are working"""
    QUICK_VERIFY_URL = "http://www.example.com"
    VERIFICATION_RECORDS = VerificationRecords()

    @classmethod
    def verify_single(cls , proxy_url: str, target_url: str , * , verify: bool = False, timeout: float = 10.0) -> bool:
        """Whether the proxy can access the test URL (GET, non-4xx/5xx is considered available)."""
        record = cls.VERIFICATION_RECORDS.get_record(target_url, proxy_url)
        if isinstance(record, bool):
            return record
        with record:
            status = test_connection(target_url, proxy_url, verify, timeout)
            record.set_status(status)
        return status

    @classmethod
    def dummy_single(cls , proxy_url: str, target_url: str , *args , timeout: float = 10.0 , **kwargs) -> bool:
        record = cls.VERIFICATION_RECORDS.get_record(target_url, proxy_url)
        if isinstance(record, bool):
            return record
        with record:
            with http_session(proxy=proxy_url,trust_env=False,verify=False,timeout=timeout):
                wait_time = random.uniform(0, timeout) + 0.2
                time.sleep(wait_time)
                check_time = random.normalvariate(timeout * 0.2 , timeout * 0.3) + 0.2
                status = wait_time < check_time
            record.set_status(status)
        return status

    @classmethod
    def stats(cls) -> pd.DataFrame:
        return cls.VERIFICATION_RECORDS.stats()

    @classmethod
    def parallel_verification(cls , proxies: Iterable[str] , target_url : str , verify: bool = False, timeout : float = 10.0 , workers : int = 50 , dummy: bool = False) -> list[str]:
        cands = dict.fromkeys(proxies , False)
        workers = max(1, min(workers, len(cands)))
        single = cls.verify_single if not dummy else cls.dummy_single
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_map = {pool.submit(single, p , target_url=target_url, verify=verify, timeout=timeout): p for p in cands}
            for fut in as_completed(fut_map):
                if fut.result():
                    cands[fut_map[fut]] = True
        return [proxy for proxy , valid in cands.items() if valid]

    @classmethod
    def verified_proxies(cls , proxies: list[str], target_url: str , * , timeout: float = 10.0, workers: int = 50, silent: bool = False, dummy: bool = False, prefix: str = '') -> list[str]:
        """parallel verify, take the first ``max_keep`` passed proxies in the original order of the candidate list."""
        if not proxies:
            return []
        with Logger.Timer(f'{prefix}Quick Verify ({timeout/2:.1f}s) for {cls.QUICK_VERIFY_URL}', indent = 1, vb_level = 3 , silent=silent) as timer:
            passed_proxies = cls.parallel_verification(proxies,cls.QUICK_VERIFY_URL,False,timeout/2,workers,dummy=dummy)
            timer.add_key_suffix(f', {len(proxies)} in {len(passed_proxies)} passed')
        if not passed_proxies:
            return []
        with Logger.Timer(f'{prefix}Final Verify ({timeout:.1f}s) for {target_url}', indent = 1, vb_level = 3 , silent=silent) as timer:
            final_proxies = cls.parallel_verification(passed_proxies,target_url,True,timeout,workers,dummy=dummy)
            timer.add_key_suffix(f', {len(passed_proxies)} in {len(final_proxies)} passed')
        return final_proxies

    @classmethod
    def get_working_proxies(
        cls , target_url: str , min_count: int = 5, * , max_round: int = 3 , timeout: float = 10.0, workers: int = 50 , dummy: bool = False, go_with_cached_proxies: bool = False) -> list[str]:
        """
        return the list of available proxy URLs; with in-process short-term cache to avoid hitting the free proxy site for each failed task.
        if force_refresh is True, ignore the expired cache.
        """
        from src.proj.util.proxy.finder import FreeProxyFinder as Finder
        from src.proj.util.proxy.cache import ProxyCache as Cache
        Logger.info(f'Get working proxies for {target_url}' + (f' with dummy verification' if dummy else ''))
        finder = Finder()
        for _ in range(2):
            with Logger.Timer(f'Round 0 Load Proxies', indent = 1, vb_level = 3 , silent=False) as timer:
                initial_proxies = Cache.get_cached_proxies('all')
                timer.add_key_suffix(f', {len(initial_proxies)} loaded')
            if go_with_cached_proxies:
                return initial_proxies
            verified_proxies = cls.verified_proxies(initial_proxies, target_url, timeout=timeout,  workers=workers , silent=False , dummy=dummy , prefix = 'Round 0 ')
            for round in range(max_round):
                if len(verified_proxies) >= min_count:
                    break
                with Logger.Timer(f'Round {round+1} Find Proxies', indent = 1, vb_level = 3 , silent=False) as timer:
                    cands = finder.find_candidates()
                    timer.add_key_suffix(f', {len(cands)} found')
                if round == 0:
                    cands = initial_proxies + cands
                cands = [c for c in cands if c not in verified_proxies]
                verified_proxies += cls.verified_proxies(cands, target_url, timeout=timeout,  workers=workers, dummy=dummy , prefix = f'Round {round+1} ')
            if verified_proxies:
                break
        if not dummy and verified_proxies:
            Cache.update(target_url, verified_proxies)
        return verified_proxies