import functools , time , random
import portalocker

from datetime import datetime
from typing import Callable

from src.proj import PATH , Logger

__all__ = ['ScriptLock' , 'ScriptLockMultiple']
class ScriptLock:
    LOCK_DIR = PATH.runtime.joinpath('script_lock')
    
    def __init__(self, lock_name: str | None = None, timeout: int | None = None , wait_time: int = 1 , vb_level : int = 1):
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
        self.vb_level = vb_level
        self.LOCK_DIR.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        """printing out message"""
        Logger.debug(message , vb_level = self.vb_level)

    def _log_get_lock(self , start_time: datetime | None = None):
        if start_time is not None and (wait_time := (datetime.now() - start_time).total_seconds()) > 1:
            self._log(f"Wait {wait_time:.1f} seconds to get lock of {self.lock_name}, continue to run.")
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
        start_time = datetime.now()         
        if self.timeout:
            while (datetime.now() - start_time).total_seconds() < self.timeout:
                try:
                    portalocker.lock(self.lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                    self._log_get_lock(start_time)
                    return self
                except (portalocker.AlreadyLocked, BlockingIOError):
                    if self.wait_time > self.timeout: 
                        raise BlockingIOError(f"Other instance is running")
                    self._log_wait_lock()
                    time.sleep(self.wait_time)
            # self._log(f"Wait {(datetime.now() - start_time).total_seconds():.1f} seconds to get lock, exceed timeout ({self.timeout} seconds)")
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

class ScriptLockMultiple:
    LOCK_DIR = PATH.runtime.joinpath('script_lock_multiple')
    
    def __init__(self, lock_name: str, lock_num: int = 1, timeout: int | None = None, 
                 wait_time: int = 1, vb_level : int = 1):
        """
        initialize multiple lock manager
        Args:
            lock_name: base name of locks
            lock_num: maximum number of instances allowed to run simultaneously
            timeout: timeout seconds, None means infinite wait
            wait_time: wait time seconds between each check
            vb_level: minimum verbosity level to output logs
        """
        self.lock_name = lock_name
        self.lock_num = lock_num
        self.timeout = timeout
        self.wait_time = wait_time
        self.vb_level = vb_level
        self.acquired_locks = []  # 存储已获取的锁文件
        self._has_wait_message = False
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @property
    def lock_dir(self):
        return self.LOCK_DIR.joinpath(f"{self.lock_name}")

    def _log(self, message: str):
        """printing out message"""
        Logger.debug(message , vb_level = self.vb_level)

    def _log_get_lock(self, lock_id: int, start_time: datetime | None = None):
        """record the log of getting lock"""
        if start_time is not None and (wait_time := (datetime.now() - start_time).total_seconds()) > 1:
            self._log(f"Wait {wait_time:.1f} seconds to get lock {lock_id} of {self.lock_name}, continue to run.")
        else:
            self._log(f"Get lock {lock_id} of {self.lock_name}, continue to run.")

    def _log_wait_lock(self):
        """record the log of waiting lock"""
        if not self._has_wait_message:
            self._log(f"All {self.lock_num} instances of {self.lock_name} are running, wait for a lock to be released...")
            self._has_wait_message = True

    def _get_lock_paths(self):
        """get all lock file paths"""
        return [self.lock_dir.joinpath(f"instance.{i}.lock") for i in range(self.lock_num)]

    def _try_acquire_any_lock(self):
        """try to acquire any available lock"""
        lock_paths = self._get_lock_paths()
        random.shuffle(lock_paths)  # shuffle the order to avoid always trying the same lock
        
        for lock_path in lock_paths:
            try:
                lock_file = open(lock_path, 'w')
                portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
                self.acquired_locks.append(lock_file)
                return int(lock_path.stem.split('.')[-1])  # return the index of the lock
            except (portalocker.AlreadyLocked, BlockingIOError):
                if 'lock_file' in locals():
                    lock_file.close()
                continue
        return None

    def __enter__(self):
        """enter context to get lock"""
        if self.lock_num <= 0:
            return self
            
        start_time = datetime.now()
        self._has_wait_message = False
        
        if self.timeout:
            # there is timeout
            while (datetime.now() - start_time).total_seconds() < self.timeout:
                lock_id = self._try_acquire_any_lock()
                if lock_id is not None:
                    self._log_get_lock(lock_id, start_time)
                    return self
                
                if self.wait_time > self.timeout:
                    raise BlockingIOError(f"All {self.lock_num} instances are running")
                
                self._log_wait_lock()
                time.sleep(self.wait_time)
            
            raise TimeoutError(f"Get lock timeout ({self.timeout} seconds) for {self.lock_name}")
        else:
            # there is no timeout
            while True:
                lock_id = self._try_acquire_any_lock()
                if lock_id is not None:
                    self._log_get_lock(lock_id)
                    return self
                
                self._log_wait_lock()
                time.sleep(self.wait_time)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """exit context to release lock"""
        for lock_file in self.acquired_locks:
            try:
                portalocker.unlock(lock_file)
                lock_file.close()
            except Exception as e:
                self._log(f"Error releasing lock: {e}")
        self.acquired_locks.clear()

    def __call__(self, func: Callable) -> Callable:
        """decorator usage"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    @classmethod
    def get_available_count(cls, lock_name: str) -> int:
        """get the number of available locks"""
        lock_dir = cls.LOCK_DIR
        if not lock_dir.exists():
            return 0
        
        available = 0
        for lock_file in lock_dir.glob(f"{lock_name}_*.lock"):
            try:
                with open(lock_file, 'r') as f:
                    portalocker.lock(f, portalocker.LOCK_EX | portalocker.LOCK_NB)
                    available += 1
                    portalocker.unlock(f)
            except (portalocker.AlreadyLocked, BlockingIOError):
                continue
        return available