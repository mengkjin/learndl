import functools , time , datetime
import portalocker

from typing import Callable

from src.proj import PATH , Logger

class ScriptLock:
    LOCK_DIR = PATH.runtime.joinpath('script_lock')
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    
    def __init__(self, lock_name: str | None = None, timeout: int | None = None , wait_time: int = 1 , verbose: bool = True):
        """
        init script lock
        Args:
            lock_name: lock file name , default is None (will not use lock)
            timeout: timeout seconds , None means infinite wait
            wait_time: wait time seconds between each check
            ** if wait_time > timeout, will only check once
        """
        self.lock_name = lock_name
        self.timeout = timeout
        self.wait_time = wait_time
        self.lock_file = None
        self.verbose = verbose

    def _log(self, message: str):
        """print out message"""
        if self.verbose:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            Logger.info(f"[{timestamp}] {message}")

    def _log_get_lock(self , start_time: float | None = None):
        if start_time is not None and time.time() - start_time > 1:
            self._log(f"Wait {time.time() - start_time:.1f} seconds to get lock of {self.lock_name}, continue to run.")
        else:
            self._log(f"Get lock of {self.lock_name}, continue to run.")

    def _log_wait_lock(self):
        if not self._has_wait_message:
            self._log(f"Other instance of {self.lock_name} is running, wait for lock to be released...")
            self._has_wait_message = True
    
    def _get_lock_path(self):
        """get lock file path"""
        if self.lock_name:
            return self.LOCK_DIR.joinpath(f"{self.lock_name}.lock")
        else:
            return None
    
    def __enter__(self):
        """enter context to get lock"""
        if (lock_path := self._get_lock_path()) is None:
            return self
        self.lock_file = open(lock_path, 'w')
        self._has_wait_message = False
        start_time = time.time()         
        if self.timeout:
            while time.time() - start_time < self.timeout:
                try:
                    portalocker.lock(self.lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                    self._log_get_lock(start_time)
                    return self
                except (portalocker.AlreadyLocked, BlockingIOError):
                    if self.wait_time > self.timeout: raise BlockingIOError(f"Other instance is running")
                    self._log_wait_lock()
                    time.sleep(self.wait_time)
            # self._log(f"Wait {time.time() - start_time:.1f} seconds to get lock, exceed timeout ({self.timeout} seconds)")
            raise TimeoutError(f"get lock timeout ({self.timeout} seconds): {lock_path}")
        else:
            try:
                portalocker.lock(self.lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                self._log_get_lock()
            except (portalocker.AlreadyLocked, BlockingIOError):
                
                self._log_wait_lock()
                portalocker.lock(self.lock_file, portalocker.LOCK_EX)
                self._log_get_lock(start_time)
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """exit context to release lock"""
        if self.lock_file:
            portalocker.unlock(self.lock_file)
            self.lock_file.close()
    
    def __call__(self, func: Callable) -> Callable:
        """decorator usage"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
