"""Reusable numeric helpers: ``basic`` (indices/fill), ``linalg``, ``tensor``, ``transform``, ``metric``.

``from func import *`` re-exports ``basic`` symbols; submodules are available as ``func.tensor``, etc.
"""

from . import (
    linalg , tensor , transform , metric
)
from .basic import *  # noqa: F403
