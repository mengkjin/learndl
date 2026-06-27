"""Configuration for boost validation metrics during training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

__all__ = ['BoostValidMetricSpec', 'ValidMetricName', 'GlbMetricName']

ValidMetricName: TypeAlias = Literal['global2top', 'rankic', 'top5pct']
GlbMetricName: TypeAlias = Literal['rankic', 'ic']


@dataclass(slots=True)
class BoostValidMetricSpec:
    """Validation metric spec for early stopping and Optuna.

    Attributes:
        name:           Primary metric name returned to boosters.
        glb_multiplier: Weight on global (RankIC/IC) term in ``global2top``.
        top_quantile:   Top fraction for the portfolio-style term (default 5%).
        glb_climax:     When set, penalise global score below this level.
        glb_metric:     Global term: ``rankic`` or ``ic``.
    """
    name: ValidMetricName = 'global2top'
    glb_multiplier: float = 5.0
    top_quantile: float = 0.05
    glb_climax: float | None = None
    glb_metric: GlbMetricName = 'rankic'

    @classmethod
    def from_dict(cls, param: dict | None) -> BoostValidMetricSpec | None:
        if not param:
            return None
        known = {k: v for k, v in param.items() if k in cls.__dataclass_fields__}
        return cls(**known)
