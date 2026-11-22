"""
Basic metaclasses for the project
"""
from __future__ import annotations
from src.shared.singleton import SingletonMeta as Singleton
from src.shared.singleton import SingletonABCMeta as SingletonABC
from src.shared.singleton import NoInstanceMeta as NoInstance

__all__ = ['Singleton' , 'SingletonABC' , 'NoInstance']

