"""Opt-in callback helpers. No global replacement of training metrics."""
from __future__ import annotations
import pandas as pd
from src.func.tensor import rank, rank_scatter
from src.func.metric import pearson

FLAGS = ('summary_latest', 'summary_hidden_rank', 'decision_local', 'basic_test', 'dispatch')
DEFAULTS = {**dict.fromkeys(FLAGS, False), 'hidden_rank_threshold': 5000}


def options(config):
    params = getattr(config, 'Param', {})
    prefix = 'train.experimental_callback_fastpath'
    values = params.get(prefix, {})
    result = DEFAULTS | (values or {})
    result.update({key: params[f'{prefix}.{key}'] for key in DEFAULTS if f'{prefix}.{key}' in params})
    if any(type(result[k]) is not bool for k in FLAGS):
        raise ValueError('callback fastpath switches must be booleans')
    if type(result['hidden_rank_threshold']) is not int or result['hidden_rank_threshold'] < 1:
        raise ValueError('hidden_rank_threshold must be a positive integer')
    return result


def latest(attempt, dataset, metric):
    tables = attempt.tables[f'{dataset}_epoch_totals']
    col = 'rankic' if metric == 'rankic' else f'total_{metric}'
    return tables[-1][col].item() if tables else 0.


class HybridSpearman:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, pred, label, weight=None, dim=None, **kwargs):
        def ranked(x):
            length = x.numel() if dim is None or x.ndim == 0 else x.shape[dim]
            return rank_scatter(x, dim=dim) if length >= self.threshold else rank(x, dim=dim)
        return pearson(ranked(pred), ranked(label), weight, dim)


def basic_group_stats(frame):
    """Same finite-row population, pandas average ties and sample std as legacy."""
    pred_rank = frame['pred'].rank()
    pct = pred_rank / frame['pred'].count()
    rtn = frame['rtn_lag1_10']
    standardized = (rtn - rtn.mean()) / (rtn.std() + 1e-6)
    return pd.Series({
        'rankic_rtn': pred_rank.corr(rtn.rank()),
        'rankic_std': pred_rank.corr(frame['std_lag1_10'].rank()),
        'top5pct': standardized.loc[pct >= .95].mean(),
        'mid20pct': standardized.loc[(pct >= .4) & (pct < .6)].mean(),
        'bot5pct': standardized.loc[pct < .05].mean(),
    })
