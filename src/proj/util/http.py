import httpx
import ssl
import sys
import time
from contextlib import contextmanager
from typing import Union , Iterable , Generator , TypeVar
from curl_cffi import requests

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
    timeout: float | tuple[float, float] | None = None,
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
    session.timeout = (30.0, 300.0) if timeout is None else timeout # (connect, read)
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
def temporary_timeout_expand(session : requests.Session, expansion : float = 2.):
    old_timeout = session.timeout
    if isinstance(old_timeout, tuple):
        new_timeout = (old_timeout[0] * expansion, old_timeout[1] * expansion)
    else:
        new_timeout = old_timeout * expansion
    session.timeout = new_timeout
    try:
        yield
    finally:
        session.timeout = old_timeout

def test_connection(target_url: str, proxy: str | None = None, timeout : float = 10. , fast_test: bool = False) -> bool:
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
    last_time = time.time()
    for i , item in enumerate(iterable):
        if i > 0 and (cost := time.time() - last_time) < interval:
            time.sleep(interval - cost)
        last_time = time.time()
        yield item
        
