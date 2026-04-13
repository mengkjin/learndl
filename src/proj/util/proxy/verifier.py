"""Parallel HEAD/GET checks against target sites; records timings and success rates."""

import time
import random
import threading
import pandas as pd
from curl_cffi import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

from src.proj.log import Logger
from src.proj.util.web import http_session , test_connection

from .core import Proxy , ProxySet

@dataclass
class VerifyRecord:
    """Timing and outcome record for one proxy-vs-target-URL verification attempt."""

    target_url: str
    proxy_url: str
    status: bool = False
    start_time: float = 0
    end_time: float = 0

    def __enter__(self):
        """Start timing the verification."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing the verification."""
        self.end_time = time.time()

    def set_status(self, status: bool):
        """Record the pass/fail outcome of this verification."""
        self.status = status

    @property
    def duration(self) -> float:
        """Wall-clock seconds taken by this verification attempt."""
        return self.end_time - self.start_time

    @property
    def info(self) -> str:
        """One-line human-readable summary: target url, proxy url, status, duration."""
        return f'{self.target_url} {self.proxy_url} {self.status} {self.duration}'

    def to_dict(self) -> dict:
        """Serialise to a dict suitable for DataFrame construction."""
        return {
            'target_url': self.target_url,
            'proxy_url': self.proxy_url,
            'status': self.status,
            'duration': self.duration,
        }

class VerificationRecords:
    """Thread-safe registry of all proxy verification attempts, keyed by (target_url, proxy_url)."""

    def __init__(self):
        self.records : dict[tuple[str, str], VerifyRecord] = {}
        self.lock = threading.Lock()

    def get_record(self, target_url: str, proxy_url: str) -> VerifyRecord | bool:
        """
        Return an existing boolean result if already verified, or create a new VerifyRecord.

        Returns a bool when the outcome is already known; returns a VerifyRecord context-manager
        when this is the first attempt for the (target_url, proxy_url) pair.
        """
        status = self.get_status(target_url, proxy_url)
        if status is not None:
            return status
        record = VerifyRecord(target_url, proxy_url)
        with self.lock:
            self.records[(target_url, proxy_url)] = record
        return record

    def get_status(self, target_url: str, proxy_url: str) -> bool | None:
        """Return the cached pass/fail result for a pair, or None if not yet verified."""
        with self.lock:
            if (target_url, proxy_url) in self.records:
                return self.records[(target_url, proxy_url)].status
            return None

    def unverified_proxies(self, target_url: str , proxies: Iterable[Proxy | str]) -> ProxySet:
        """Filter ``proxies`` to those not yet attempted for ``target_url``."""
        with self.lock:
            return ProxySet([proxy for proxy in proxies if (target_url, str(proxy)) not in self.records])

    def stats(self) -> pd.DataFrame:
        """Return a DataFrame summarising verification counts and timings grouped by target and status."""
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
    def verify_single(cls , proxy: Proxy | str , target_url: str , * , timeout: float = 10.0 , fast_test: bool = False) -> bool:
        """Whether the proxy can access the test URL (GET, non-4xx/5xx is considered available)."""
        record = cls.VERIFICATION_RECORDS.get_record(target_url, str(proxy))
        if isinstance(record, bool):
            return record
        with record:
            status = test_connection(target_url, str(proxy), timeout , fast_test=fast_test)
            record.set_status(status)
            if status and isinstance(proxy, Proxy):
                proxy.verified.append(target_url)
        return status

    @classmethod
    def unverified_proxies(cls, target_url: str , proxies: Iterable[Proxy | str]) -> ProxySet:
        """Return the subset of ``proxies`` that have not yet been tested against ``target_url``."""
        return cls.VERIFICATION_RECORDS.unverified_proxies(target_url, proxies)

    @classmethod
    def dummy_single(cls , proxy: Proxy | str, target_url: str , *args , timeout: float = 10.0 , **kwargs) -> bool:
        """Simulate a verification with random pass/fail after a random sleep — used for testing."""
        record = cls.VERIFICATION_RECORDS.get_record(target_url, str(proxy))
        if isinstance(record, bool):
            return record
        with record:
            with http_session(proxy=str(proxy),trust_env=False,verify=False,timeout=timeout):
                wait_time = random.uniform(0, timeout) + 0.2
                time.sleep(wait_time)
                check_time = random.normalvariate(timeout * 0.2 , timeout * 0.3) + 0.2
                status = wait_time < check_time
            record.set_status(status)
        return status

    @classmethod
    def stats(cls) -> pd.DataFrame:
        """Delegate to :meth:`VerificationRecords.stats` for the process-wide verification summary."""
        return cls.VERIFICATION_RECORDS.stats()

    @classmethod
    def parallel_verification(cls , proxies: Iterable[Proxy | str] , target_url : str , timeout : float = 10.0 , * ,
                              fast_test: bool = False, workers : int = 50 , dummy: bool = False) -> ProxySet:
        """Verify all proxies against ``target_url`` concurrently; return only those that passed."""
        cands = dict.fromkeys(proxies , False)
        if not cands:
            return ProxySet()
        workers = max(1, min(workers, len(cands)))
        single = cls.verify_single if not dummy else cls.dummy_single
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_map = {pool.submit(single, p , target_url=target_url, timeout=timeout , fast_test=fast_test): p for p in cands}
            for fut in as_completed(fut_map):
                if fut.result():
                    cands[fut_map[fut]] = True
        return ProxySet([proxy for proxy , valid in cands.items() if valid])

    @classmethod
    def verified_proxies(cls , proxies: Iterable[Proxy | str] , target_url: str , * , timeout: float = 10.0, workers: int = 50, silent: bool = False, dummy: bool = False) -> ProxySet:
        """parallel verify, take the first ``max_keep`` passed proxies in the original order of the candidate list."""
        proxies = ProxySet(proxies)
        with Logger.Timer(f'Quick Verify ({timeout/2:.1f}s) for {cls.QUICK_VERIFY_URL}', indent = 1, vb_level = 3 , silent=silent) as timer:
            passed_proxies = cls.parallel_verification(proxies,cls.QUICK_VERIFY_URL,timeout,fast_test=True,workers=workers,dummy=dummy)
            timer.add_key_suffix(f', {len(passed_proxies)}/{len(proxies)} passed')
        with Logger.Timer(f'Final Verify ({timeout:.1f}s) for {target_url}', indent = 1, vb_level = 3 , silent=silent) as timer:
            final_proxies = cls.parallel_verification(passed_proxies,target_url,timeout,fast_test=False,workers=workers,dummy=dummy)
            timer.add_key_suffix(f', {len(final_proxies)}/{len(passed_proxies)} passed')
        return final_proxies

    @classmethod
    def get_real_ip(cls, timeout: float = 5.0) -> str:
        """Get real public IP without using proxy"""
        try:
            resp = requests.get("https://httpbin.org/ip", timeout=timeout)
            return resp.json().get("origin", "")
        except Exception as e:
            Logger.stderr(f"Failed to get real public IP: {e}")
            return ""

    @classmethod
    def check_proxy_anonymity(cls, proxy: Proxy | str, real_ip: str , test_url: str = "https://httpbin.org/ip", timeout: float = 5.0) -> Proxy | None:
        """Check proxy anonymity (whether transparent)"""
        try:
            resp = requests.get(test_url, proxy=str(proxy), timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            proxy_ip = data.get("origin", "")
            headers = resp.headers
            via = headers.get("Via", "")
            x_forwarded_for = headers.get("X-Forwarded-For", "")

            is_transparent = False
            # if the proxy returned IP is equal to the real IP, it means it is a transparent proxy (or direct connection)
            if proxy_ip == real_ip:
                is_transparent = True
            # if the response header has X-Forwarded-For and contains the real IP, it is also considered a transparent proxy
            elif x_forwarded_for and real_ip in x_forwarded_for:
                is_transparent = True
            # if the response header has Via field, it usually means there is a proxy in the middle (not necessarily transparent, but indicates non-anonymous)
            elif via:
                is_transparent = True

            if is_transparent:
                return None
            else:
                return Proxy(proxy)
        except Exception:
            return None

    @classmethod
    def filter_proxies_by_anonymity(cls, proxies: Iterable[Proxy | str]) -> ProxySet:
        """Remove transparent proxies: fetch the real public IP, then drop any proxy that leaks it."""
        real_ip = cls.get_real_ip()
        if not real_ip:
            Logger.alert2("Failed to get real public IP, cannot check proxy anonymity")
            return ProxySet()

        proxies = ProxySet(proxies)
        anonymized_proxies = dict.fromkeys(proxies , False)
        with ThreadPoolExecutor(max_workers=50) as pool:
            fut_map = {pool.submit(cls.check_proxy_anonymity, proxy, real_ip): proxy for proxy in proxies}
            for fut in as_completed(fut_map):
                if fut.result():
                    anonymized_proxies[fut_map[fut]] = True
        return ProxySet([proxy for proxy , valid in anonymized_proxies.items() if valid])
