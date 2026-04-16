"""Algorithm sub-package: NN and gradient-boost model registry.

Re-exports :class:`AlgoModule`, which provides class-method factories
``get_nn`` / ``get_boost`` and utilities for listing available models.
"""
from .api import AlgoModule