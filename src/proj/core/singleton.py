"""Thread-safe singleton decorator and metaclasses."""

import threading
from abc import ABCMeta

def singleton(cls):
    '''thread safe singleton decorator'''
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]

    return get_instance

class NoInstanceMeta(type):
    '''metaclass to block direct instantiation of a class'''
    def __new__(cls, name, bases, namespace, **kwargs):
        return super().__new__(cls, name, bases, namespace)
    def __call__(cls, *args, **kwargs):
        raise Exception(f'Class {cls.__name__} should not be called to create instance')

class SingletonMeta(type):
    """First ``__call__`` constructs the instance; later calls return the same object (thread-safe)."""

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]

class SingletonABCMeta(SingletonMeta, ABCMeta):
    """``SingletonMeta`` composed with ``ABCMeta`` for abstract singleton bases."""

    def __new__(mcls, name, bases, namespace, **kwargs):
        return super().__new__(mcls, name, bases, namespace)
