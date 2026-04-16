"""Public API surface for the boost sub-package.

Re-exports:
    ``AVAILABLE_BOOSTS``   — dict mapping name → class for all registered boosters.
    :class:`GeneralBoostModel` — unified booster front-end.
    :class:`OptunaBoostModel`  — Optuna HPO variant.
    :class:`BoostInput`        — aligned 3-D data container.
    :class:`BoostOutput`       — flat prediction container with index.
"""
from .util import BoostInput , BoostOutput
from .booster import GeneralBoostModel , OptunaBoostModel , AVAILABLE_BOOSTS

__all__ = ['AVAILABLE_BOOSTS' , 'GeneralBoostModel' , 'OptunaBoostModel' , 'BoostInput' , 'BoostOutput']