import httpx
import time
from curl_cffi.requests import exceptions
from typing import Callable , Literal , TypeVar , Generic , Any , Type , Union , Tuple
from src.proj.log import Logger

T = TypeVar("T")
TInside = TypeVar('TInside')

class UnattainableException(Exception):
    ...

def get_exception_name(e: Exception) -> str:
    exc_type = type(e)
    return f"{exc_type.__module__}.{exc_type.__qualname__}"

def retry_on_exceptions(func: Callable[..., T], error_handler: Callable[[Exception], str | Exception] , * , label: str, attempts: int = 1, base_delay: float = 1.5) -> Callable[..., T | Exception]:
    """
    Retry on exceptions with a given error handler
    """
    assert attempts > 0 and base_delay >= 0 , f"attempts {attempts} and base_delay {base_delay} invalid"
    def wrapper(*args, **kwargs) -> T | Exception:
        for i in range(attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ramified_error = error_handler(e)
                if isinstance(ramified_error, Exception):
                    raise ramified_error
                if i == attempts - 1:
                    Logger.alert1(f"{label} >> {ramified_error} Reached maximum retry attempts, giving up: {e!s}")
                    return e
                else:
                    delay = base_delay * (2**i)
                    Logger.skipping(f"{label} >> {ramified_error} will retry in {delay:.1f}s ({i + 1}/{attempts}): {e!s}")
                    time.sleep(delay)
        else:
            return UnattainableException(f"Unattainable Exception: reached maximum retry attempts ({attempts}), giving up")
    return wrapper

def exception_handler(e : Exception , exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception) -> str | Exception:
    if isinstance(e, exceptions):
        return f"{e!s}"
    return e

def http_handler(e: Exception) -> str | Exception:
    if isinstance(e, (httpx.HTTPStatusError , exceptions.HTTPError , httpx.HTTPError , exceptions.RequestException)):
        if isinstance(e, (httpx.HTTPStatusError , exceptions.HTTPError)) and e.response is not None and e.response.status_code in (429, 502, 503, 504):
            return f"Bad HTTP response status code [{e.response.status_code}]"
        else:
            return f"HTTP exception [{get_exception_name(e)}]"
    return e

def all_handler(e: Exception) -> str | Exception:
    return f"Unhandled exception [{get_exception_name(e)}]"

def retry_call(
    func: Callable[..., T], args: tuple = (), kwargs: dict | None = None,
    attempts: int = 3,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    base_delay: float = 1.5,
    error: Literal['raise' , 'return'] = 'raise',
) -> T | Exception:
    """
    retry a function call with a given exceptions and delay
    """
    kwargs = kwargs or {}
   
    for i in range(attempts):
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            if i == attempts - 1:
                if error == 'raise':
                    raise
                else:
                    return e
            delay = base_delay * (2**i)
            Logger.skipping(f"Attempt {i + 1} / {attempts} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    else:
        return UnattainableException(f"Unattainable Exception: reached maximum retry attempts ({attempts}), giving up")

class ErrorHandler(Generic[T]):
    """
    Error handler for function calls

    return a WrappedResult object with the following properties:
        - success: bool
        - is_http_error: bool
        - error_type: str
        - unwrap: callable
            - error: 'raise' | 'return'
            - return the raw result or raise the exception
    example:
        result = ErrorHandler(func, catch_errors={'http' : {'attempts' : 2 , 'base_delay' : 1.5}}, label='func')
        result.unwrap(error='return')
        result.unwrap(error='raise')
        result.success
        result.is_http_error
        result.error_type
    """
    DEFAULT_KWARGS = {
        'http' : {'attempts' : 2 , 'base_delay' : 1.5},
        'all' : {'attempts' : 1 , 'base_delay' : 0},
    }
    def __init__(
        self, func: Callable[..., T | Exception] , 
        handle_types: dict[str , dict[str,Any]] | list[str] | tuple[str,...] = () , * , label: str = ''):
        self.raw_func = func
        for error_type in handle_types:
            if isinstance(handle_types, dict):
                kwargs = handle_types.get(error_type, {})
            else:
                kwargs = self.DEFAULT_KWARGS.get(error_type, {})
            func = retry_on_exceptions(func, self.handler(error_type), label=label, **kwargs)
        self.wrapped_func = func
        self.label = label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.wrapped_func})"

    def __call__(self, *args, **kwargs) -> 'WrappedResult':
        raw_result = self.wrapped_func(*args, **kwargs)
        return self.WrappedResult(raw_result)

    def handler(self, handler_type: str) -> Callable[[Exception], str | Exception]:
        match handler_type:
            case 'http':
                return http_handler
            case 'all':
                return all_handler
            case _:
                raise ValueError(f"Invalid handler type: {handler_type}")

    class WrappedResult(Generic[T]): # type: ignore
        def __init__(self, raw_result: 'T | Exception'):
            self.raw_result = raw_result

        @property
        def success(self) -> bool:
            return not isinstance(self.raw_result, Exception)

        @property
        def is_http_error(self) -> bool:
            return isinstance(self.raw_result, (httpx.HTTPStatusError, exceptions.HTTPError , httpx.HTTPError , exceptions.RequestException))

        @property
        def error_type(self) -> str:
            if not isinstance(self.raw_result, Exception):
                return ''
            elif self.is_http_error:
                return 'http'
            else:
                return type(self.raw_result).__name__

        def unwrap(self , error : Literal['raise' , 'return'] = 'return') -> 'T | Exception':
            if isinstance(self.raw_result, Exception) and error == 'raise':
                raise self.raw_result
            return self.raw_result