"""Opt-in metric fast path. Legacy classes remain the default and reference implementation."""
from __future__ import annotations

import pandas as pd
import torch

from src.res.algo.nn.loss import Accuracy
from .functions import AccuracyFunction
from .stage_metrics import BatchMetrics, EpochMetrics, AggregatedMetrics


class ReusingAccuracyFunction(AccuracyFunction):
    """Resolve equivalence once after effective components (including net overrides) exist."""

    def __init__(self, *args, model_param, label_count, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_rankic = None
        component = self.components.get('spearman')

        def first_only(selection):
            if selection is None:
                return True  # Only used with a proven single output/label below.
            if isinstance(selection, int):
                return selection == 0
            return list(selection) == [0]

        self.reuse_rankic = False
        if tuple(self.components) == ('spearman',) and component is not None:
            output = model_param.get('which_output')
            label = model_param.get('which_label')
            output = component.which_output if output is None else output
            label = component.which_label if label is None else label
            self.reuse_rankic = (
                model_param.get('num_output', 1) == 1 and label_count == 1
                and type(component.calculator) is type(Accuracy.get('spearman'))
                and component.lamb == 1.0 and not component.kwargs
                and self.ExcludeNan and first_only(output) and first_only(label)
            )

    def accuracies(self, data, which_output=None, which_label=None, dataset='train'):
        self.last_rankic = None
        if not self.reuse_rankic or dataset not in ('train', 'valid'):
            return super().accuracies(data, which_output, which_label, dataset)
        values = self(data, which_output=which_output, which_label=which_label, require_grad=False)
        # Keep the original immediate failure and scalar-read timing.
        if any(torch.isnan(value) for value in values.values()):
            bad = {key: float(value) for key, value in values.items() if torch.isnan(value)}
            raise ValueError(f'Encountered nan accuracies! {bad}')
        result = {key: value.item() if isinstance(value, torch.Tensor) else value
                  for key, value in values.items()}
        self.last_rankic = values['spearman'].mean()
        return result


class FastBatchMetrics(BatchMetrics):
    """Cache only already-requested scalar values; never postpone the first read."""

    def __init__(self, aggregator):
        self._scalar_cache = {}
        super().__init__(aggregator)

    def set_metrics(self, key, metrics):
        if key == 'losses':
            self._scalar_cache.pop('losses', None)
        elif key == 'rankic':
            self._scalar_cache.pop('rankic', None)
        super().set_metrics(key, metrics)

    def reset_metrics(self, key):
        self._scalar_cache.clear()
        super().reset_metrics(key)

    @property
    def losses_items(self):
        assert self.initiated, 'cannot access losses items when BatchMetrics is not initiated'
        if 'losses' not in self._scalar_cache:
            self._scalar_cache['losses'] = super().losses_items
        return dict(self._scalar_cache['losses'])

    @property
    def loss_weights(self):
        # The aggregator needs keys only; keep weights live for Global2Top.
        return self.aggregator.loss_weights(self.losses)

    @property
    def table_weights(self):
        return pd.DataFrame(self.loss_weights, index=self.table_index)

    def scalar_totals(self):
        if 'rankic' not in self._scalar_cache:
            self._scalar_cache['rankic'] = self.rankic.item()
        return dict(rankic=self._scalar_cache['rankic'], total_loss=self.total_loss_item,
                    total_accuracy=self.total_accuracy)

    @property
    def table_total(self):
        return pd.DataFrame(self.scalar_totals(), index=self.table_index)


class FastEpochMetrics(EpochMetrics):
    """Keep all batch records; materialize tables only on access/collection."""

    def new(self, dataset, epoch=0, phase=0, **kwargs):
        super().new(dataset, epoch, phase, **kwargs)
        self._rows = {name: [] for name in ('accuracies', 'losses', 'weights', 'totals')}
        # Preserve the collection/export table names, not the legacy internal row lists.
        for name in self._rows:
            self.tables[name] = []

    def append(self, batch: FastBatchMetrics):
        AggregatedMetrics.append(self, batch)
        self.indices['batch'].append(batch.batch_key)
        self._rows['accuracies'].append(dict(batch.accuracies))
        self._rows['losses'].append(batch.losses_items)
        self._rows['weights'].append(dict(batch.loss_weights))
        self._rows['totals'].append(batch.scalar_totals())
        self.collected_tables.clear()

    def get_table(self, name):
        if name not in self._rows:
            return super().get_table(name)
        if name not in self.collected_tables:
            rows = self._rows[name]
            if not rows:
                return pd.DataFrame()
            index = pd.MultiIndex.from_arrays(
                [[value] * len(rows) for value in self.key.values()] + [self.indices['batch']],
                names=[*self.key, 'batch'],
            )
            self.collected_tables[name] = pd.DataFrame(rows, index=index)
        return self.collected_tables[name]

    @property
    def nanloss(self):
        # Same columns and NaN (not Inf) semantics, available before epoch collection.
        return any(pd.isna(value) for row in self._rows['totals'] for value in row.values())
