"""Thread-safe singleton decorator and metaclasses."""

from abc import ABCMeta

class NoInstanceMeta(type):
    '''metaclass to block direct instantiation of a class'''
    def __new__(cls, name, bases, namespace, **kwargs):
        return super().__new__(cls, name, bases, namespace)
    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class SingletonMeta(type):
    """First ``__call__`` constructs the instance; later calls return the same object (thread-safe)."""

    _instances = {}
    _singleton_meta_lock = None

    @classmethod
    def _get_singleton_meta_lock(cls):
        """get the lock for the once"""
        from threading import Lock
        if cls._singleton_meta_lock is None:
            cls._singleton_meta_lock = Lock()
        return cls._singleton_meta_lock

    def __call__(cls, *args, **kwargs):
        with cls._get_singleton_meta_lock():
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]

class SingletonABCMeta(SingletonMeta, ABCMeta):
    """``SingletonMeta`` composed with ``ABCMeta`` for abstract singleton bases."""
    def __new__(mcls, name, bases, namespace, **kwargs):
        return super().__new__(mcls, name, bases, namespace)
