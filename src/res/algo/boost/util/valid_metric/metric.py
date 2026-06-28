"""Boost validation metric with feval adapters for gradient boosters."""
from __future__ import annotations

import numpy as np
import torch

from .scorers import aggregate_score, global2top_score, mean_global_metric, mean_top_quantile
from .spec import BoostValidMetricSpec

__all__ = ['BoostValidMetric']


class BoostValidMetric:
    """Custom validation metric for boost training (Global2Top-style)."""

    def __init__(
        self,
        spec: BoostValidMetricSpec,
        raw_label: torch.Tensor | np.ndarray,
        date: np.ndarray,
    ):
        self.spec = spec
        self.raw_label = self._as_numpy(raw_label)
        self.date = np.asarray(date)

    @staticmethod
    def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float64, copy=False)
        return np.asarray(value, dtype=np.float64)

    @classmethod
    def from_spec(
        cls,
        param: dict | None,
        *,
        raw_label: torch.Tensor | np.ndarray | None,
        date: np.ndarray | None,
    ) -> BoostValidMetric | None:
        spec = BoostValidMetricSpec.from_dict(param)
        if spec is None or raw_label is None or date is None:
            return None
        return cls(spec, raw_label, date)

    def score(self, pred: torch.Tensor | np.ndarray) -> float:
        pred_arr = self._as_numpy(pred)
        if len(pred_arr) != len(self.raw_label):
            raise ValueError(
                f'pred length {len(pred_arr)} != label length {len(self.raw_label)}; '
                'predictions must align with finite-valid rows used in BoostDataset',
            )
        return aggregate_score(pred_arr, self.raw_label, self.date, self.spec)

    @staticmethod
    def _flat_pred_from_output(output) -> torch.Tensor | np.ndarray:
        """Extract finite-aligned flat preds matching :class:`BoostDataset` rows."""
        raw_pred = getattr(output, '_raw_pred', None)
        if raw_pred is not None and int(raw_pred.numel()) == int(output.finite.sum()):
            return raw_pred
        pred = output.pred
        if pred.ndim == 2:
            return pred[output.finite]
        return pred.reshape(-1)

    @property
    def metric_name(self) -> str:
        return self.spec.name

    @property
    def higher_is_better(self) -> bool:
        return True

    def lgbm_feval(self, preds: np.ndarray, _dataset) -> tuple[str, float, bool]:
        return self.metric_name, self.score(preds), self.higher_is_better

    def xgb_feval(self, preds: np.ndarray, _dtrain):
        return self.metric_name, self.score(preds)

    def score_output(self, output) -> float:
        """Score from a :class:`BoostOutput` (post-fit / Optuna)."""
        return self.score(self._flat_pred_from_output(output))

    def build_catboost_metric(self):
        metric = self

        class _CatBoostValidMetric:
            def get_final_error(self, error, weight):
                return error

            def is_max_optimal(self):
                return True

            def evaluate(self, approxes, target, weight):
                preds = np.asarray(approxes[0], dtype=np.float64)
                return metric.score(preds), 1.0

        _CatBoostValidMetric.__name__ = f'BoostValid_{metric.metric_name}'
        return _CatBoostValidMetric()

    def decompose(self, pred: torch.Tensor | np.ndarray) -> dict[str, float]:
        pred_arr = self._as_numpy(pred)
        glb = mean_global_metric(
            pred_arr, self.raw_label, self.date, glb_metric=self.spec.glb_metric,
        )
        top = mean_top_quantile(
            pred_arr, self.raw_label, self.date, top_quantile=self.spec.top_quantile,
        )
        return {
            'global': glb,
            'top': top,
            'score': global2top_score(glb, top, self.spec),
        }
