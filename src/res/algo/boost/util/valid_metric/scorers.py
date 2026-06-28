"""Cross-sectional scorers for flat boost validation arrays."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .spec import BoostValidMetricSpec

__all__ = [
    'mean_global_metric',
    'mean_top_quantile',
    'global2top_score',
    'aggregate_score',
]


def _finite_pairs(pred: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(pred) & np.isfinite(label)
    return pred[mask], label[mask]


def _has_variance(arr: np.ndarray) -> bool:
    """True when ``arr`` has at least two distinct finite values."""
    if len(arr) < 2:
        return False
    return bool(np.nanstd(arr) > 0)


def rankic_group(pred: np.ndarray, label: np.ndarray) -> float:
    p, y = _finite_pairs(pred, label)
    if len(p) < 2 or not _has_variance(p) or not _has_variance(y):
        return float('nan')
    result = spearmanr(p, y)
    corr = float(result[0])  # type: ignore[index]
    return corr if np.isfinite(corr) else float('nan')


def ic_group(pred: np.ndarray, label: np.ndarray) -> float:
    p, y = _finite_pairs(pred, label)
    if len(p) < 2 or not _has_variance(p) or not _has_variance(y):
        return float('nan')
    result = pearsonr(p, y)
    corr = float(result[0])  # type: ignore[index]
    return corr if np.isfinite(corr) else float('nan')


def top_quantile_mean_group(pred: np.ndarray, label: np.ndarray, top_quantile: float) -> float:
    p, y = _finite_pairs(pred, label)
    if len(p) == 0:
        return float('nan')
    threshold = 1.0 - top_quantile
    ranks = np.argsort(np.argsort(p)).astype(float)
    pct = ranks / max(len(p) - 1, 1)
    top = pct >= threshold
    if not top.any():
        return float('nan')
    return float(np.nanmean(y[top]))


def mean_global_metric(
    pred: np.ndarray,
    label: np.ndarray,
    date: np.ndarray,
    *,
    glb_metric: str = 'rankic',
) -> float:
    scores: list[float] = []
    scorer = rankic_group if glb_metric == 'rankic' else ic_group
    for d in np.unique(date):
        mask = date == d
        val = scorer(pred[mask], label[mask])
        if np.isfinite(val):
            scores.append(val)
    return float(np.nanmean(scores)) if scores else float('nan')


def mean_top_quantile(
    pred: np.ndarray,
    label: np.ndarray,
    date: np.ndarray,
    *,
    top_quantile: float,
) -> float:
    scores: list[float] = []
    for d in np.unique(date):
        mask = date == d
        val = top_quantile_mean_group(pred[mask], label[mask], top_quantile)
        if np.isfinite(val):
            scores.append(val)
    return float(np.nanmean(scores)) if scores else float('nan')


def global2top_score(glb: float, top: float, spec: BoostValidMetricSpec) -> float:
    if not np.isfinite(glb):
        glb = 0.0
    if not np.isfinite(top):
        top = 0.0
    if spec.glb_climax is None:
        return top + glb * spec.glb_multiplier
    return top - max(0.0, spec.glb_climax - glb) * spec.glb_multiplier

def aggregate_score(
    pred: np.ndarray,
    label: np.ndarray,
    date: np.ndarray,
    spec: BoostValidMetricSpec,
) -> float:
    glb = mean_global_metric(pred, label, date, glb_metric=spec.glb_metric)
    top = mean_top_quantile(pred, label, date, top_quantile=spec.top_quantile)
    if spec.name == 'rankic':
        return glb
    if spec.name == 'top5pct':
        return top
    return global2top_score(glb, top, spec)
