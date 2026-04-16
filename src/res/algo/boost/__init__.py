"""Gradient-boost model sub-package.

Provides :class:`GeneralBoostModel` (dispatch wrapper for LGBM / XGBoost /
CatBoost / AdaBoost) and :class:`OptunaBoostModel` (Optuna HPO variant),
together with the :class:`BoostInput` / :class:`BoostOutput` data containers.
"""
from .util import BoostInput , BoostOutput
from .booster import  GeneralBoostModel , OptunaBoostModel
from . import api