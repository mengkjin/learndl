"""HTTP clients: ``curl_cffi`` session factory, timeout retry helpers, and ``httpx`` with sane TLS defaults."""

import httpx
import ssl
import sys
import time
from contextlib import contextmanager
from typing import Union , Iterable , Generator , TypeVar , Literal
from curl_cffi import requests

from src.proj.log import Logger

_SSLVerify = Union[str, ssl.SSLContext]
T = TypeVar("T")

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def http_session(
    *,
    proxy: str | None = None,
    trust_env: bool = False,       # curl_cffi 默认不信任环境变量
    verify: bool = True,
    allow_redirects : bool = True,
    timeout: float | tuple[float, float] = (30.0, 300.0),
    **kwargs,
) -> requests.Session:
    """
    return a curl_cffi Session, can be reused.
    Note: The Session is specified when it is created, and all subsequent requests will use this proxy.
    If you need to dynamically switch proxies, you can pass the proxies parameter in each request.
    """
    session = requests.Session()
    # set global headers
    session.headers.update({"User-Agent": CHROME_UA})
    # overall timeout (connect timeout + read timeout)
    # The timeout parameter of curl_cffi can be a (connect, read) tuple, or a single float (overall).
    session.timeout = timeout # (connect, read)
    session.allow_redirects = allow_redirects
    session.verify = verify
    session.trust_env = trust_env
    if proxy:
        session.proxies = {
            "http": proxy,
            "https": proxy,
        }
    return session

@contextmanager
def temporary_timeout_session(session : requests.Session, new_timeout : float | tuple[float, float]):
    """Temporarily set ``session.timeout``; restore previous value after the block."""
    old_timeout = session.timeout
    session.timeout = new_timeout
    try:
        yield
    finally:
        session.timeout = old_timeout

def timeout_expanding_sessions(session : requests.Session, expansion : float = 2. , max_count : int = 1):
    """
    automatically expand the timeout of the session by the given expansion factor and max count
    will return a generator of (max_count + 1) sessions with progressively longer timeouts
    - the first session has the original timeout
    - the last session has the timeout expanded by the given expansion factor ** max_count
    """
    assert expansion ** max_count < 10 , f'expansion ** max_count = {expansion ** max_count} is too large'
    old_timeout = session.timeout
    timeouts = []
    for i in range(max_count + 1):
        x = expansion ** i
        timeouts.append((old_timeout[0] * x, old_timeout[1] * x) if isinstance(old_timeout, tuple) else old_timeout * x)
    for timeout in timeouts:
        with temporary_timeout_session(session, timeout):
            yield session

def request_with_timeouterror(session: requests.Session, request_method: Literal['get', 'post'], *args, expansion : float = 2. , max_retry_count: int = 2, **kwargs) -> requests.Response:
    """GET/POST with exponentially growing timeouts until success or retries exhausted."""
    if expansion < 1:
        Logger.alert1(f"expansion {expansion} is less than 1, setting to 1")
        expansion = 1
    if max_retry_count < 1:
        Logger.alert1(f"max_retry_count {max_retry_count} is less than 1, setting to 1")
        max_retry_count = 1
    match request_method:
        case 'get':
            method = session.get
        case 'post':
            method = session.post
        case _:
            raise ValueError(f"Invalid request method: {request_method}")
    for i , _ in enumerate(timeout_expanding_sessions(session , expansion = expansion, max_count = max_retry_count)):
        try:
            r = method(*args, **kwargs)
            r.raise_for_status()
            return r
        except (TimeoutError , requests.exceptions.Timeout) as e:
            if i == max_retry_count:
                raise e
            Logger.alert1(f"requests.Session(timeout={session.timeout}) encountered TimeoutError (expand {expansion} times to retry): {e!s}")
    raise

def test_connection(target_url: str, proxy: str | None = None, timeout : float = 10. , fast_test: bool = False) -> bool:
    """Return True if HEAD returns status < 400 (optional proxy, shorter timeout when ``fast_test``)."""
    kwargs = {
        'verify': False if fast_test else True,
        'timeout': timeout / 2 if fast_test else timeout,
        'allow_redirects': False,
    }
    try:
        with http_session(proxy=proxy, trust_env=proxy is None, **kwargs) as session:
            r = session.head(target_url)
            status = r.status_code < 400
    except Exception:
        status = False
    return status

def http_client(
    *,
    proxy: str | None = None,
    trust_env: bool | None = None,
    **kwargs: object,
) -> httpx.Client:
    """Configured ``httpx.Client`` (no keep-alive, long read timeout, Chrome-like UA)."""
    # Disable keep-alive to reduce "Server disconnected" type half-open connections
    limits = httpx.Limits(max_keepalive_connections=0, max_connections=20)
    kw: dict = {
        "headers": {"User-Agent": CHROME_UA},
        "timeout": httpx.Timeout(300.0, connect=30.0),
        "follow_redirects": True,
        "limits": limits,
        "verify": default_http_verify(),
    }
    if proxy:
        kw["proxy"] = proxy
    if trust_env is not None:
        kw["trust_env"] = trust_env
    kw.update(kwargs)
    return httpx.Client(**kw)

def default_http_verify() -> _SSLVerify:
    """CA bundle path or SSLContext for httpx ``verify=``.
    On macOS, ``truststore`` uses the system keychain (fixes python.org builds with
    broken ``ssl`` defaults and trusts enterprise roots installed there). Other
    platforms use Mozilla CA via ``certifi``.
    """
    if sys.platform == "darwin":
        try:
            import truststore

            return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            pass
    import certifi
    return certifi.where()

def iterate_with_interval_control(iterable: Iterable[T], * , interval: float = 1.0) -> Generator[T, None, None]:
    """Yield items from ``iterable`` spacing iterations by at least ``interval`` seconds."""
    last_time = time.time()
    for i , item in enumerate(iterable):
        if i > 0 and (cost := time.time() - last_time) < interval:
            time.sleep(interval - cost)
        last_time = time.time()
        yield item
        
