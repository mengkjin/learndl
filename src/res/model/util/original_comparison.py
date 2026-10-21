"""Compare explicit experiment artifacts; never guess which old run is the baseline.

python -m src.res.model.util.original_comparison --baseline-log OLD.html \
    --candidate-log NEW.html --baseline-model OLD_DIR --candidate-model NEW_DIR \
    --output comparison.json
Missing artifacts/metrics remain null. Test metrics use intersecting model/date keys.
"""
from __future__ import annotations
import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LogText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def training_rows(path: Path | None):
    if path is None or not path.exists():
        return {}
    text = path.read_text()
    if path.suffix.lower() == '.html':
        parser = LogText()
        parser.feed(text)
        text = '\n'.join(parser.parts)
    number = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    pattern = (r'#(?P<model_num>\d+)\s*@(?P<model_date>\d{8})\s*\|Trial(?P<retry>\d+)-(?P<nan_redo>\d+)'
               r'\s+ModelEpoch#(?P<epochs>\d+)[^|]*\|BestAttempt\s+Trial\d+-\d+\s+'
               rf'TrainIC\s+(?P<train_ic>{number})\s+ValidIC\s+(?P<valid_ic>{number})')
    rows = {}
    for match in re.finditer(pattern, text):
        d = match.groupdict()
        key = (int(d['model_num']), int(d['model_date']))
        if key in rows:
            raise ValueError(f'Multiple training summaries for {key}; select a single run log')
        rows[key] = {name: (float(value) if name.endswith('_ic') else int(value))
                     for name, value in d.items()}
    return rows


def sample_count(root: Path | None, model_date: int):
    if root is None:
        return None
    path = root / 'snapshot' / '15m_original' / f'fit.{model_date}.json'
    return json.loads(path.read_text())['effective_count'] if path.exists() else None


def daily_results(root: Path | None):
    path = None if root is None else root / 'snapshot' / 'basic_test' / 'test_by_date.feather'
    return pd.read_feather(path) if path is not None and path.exists() else None


def compare_runs(baseline_log=None, candidate_log=None, baseline_model=None, candidate_model=None):
    old, new = training_rows(baseline_log), training_rows(candidate_log)
    output = {'missing_policy': 'null means unavailable; never inferred from other periods',
              'test_policy': 'intersection of model_num/model_date/submodel/date; finite metric pairs only',
              'top5pct_unit': 'zscore of rtn_lag1_10, not percentage return', 'models': [], 'test': []}
    for key in sorted(set(old) & set(new)):
        a, b = dict(old[key]), dict(new[key])
        a['effective_samples'] = sample_count(baseline_model, key[1])
        b['effective_samples'] = sample_count(candidate_model, key[1])
        names = ('epochs', 'retry', 'nan_redo', 'train_ic', 'valid_ic', 'effective_samples')
        output['models'].append({'model_num': key[0], 'model_date': key[1], 'baseline': a, 'candidate': b,
            'delta': {name: b[name] - a[name] if a[name] is not None and b[name] is not None else None for name in names}})
    output['training_status'] = 'compared' if output['models'] else 'unavailable: no common training summaries'
    output['baseline_only_models'] = sorted(set(old) - set(new))
    output['candidate_only_models'] = sorted(set(new) - set(old))
    a, b = daily_results(baseline_model), daily_results(candidate_model)
    if a is None or b is None:
        output['test_status'] = 'unavailable: both daily test artifacts required to align test dates'
        return output
    keys = ['model_num', 'model_date', 'submodel', 'date']
    metrics = ['rankic_rtn', 'rankic_std', 'top5pct', 'mid20pct', 'bot5pct']
    common = a.merge(b, on=keys, how='inner', suffixes=('_old', '_new'), validate='one_to_one')
    for key, group in common.groupby(keys[:3]):
        row: dict[str, Any] = dict(zip(keys[:3], key))
        row['metrics'] = {}
        for name in metrics:
            if name + '_old' not in group or name + '_new' not in group:
                row['metrics'][name] = None
                continue
            x, y = group[name + '_old'], group[name + '_new']
            valid = np.isfinite(x) & np.isfinite(y)
            row['metrics'][name] = ({'baseline': float(x[valid].mean()), 'candidate': float(y[valid].mean()),
                'delta': float((y[valid] - x[valid]).mean()), 'common_dates': group.loc[valid, 'date'].tolist()}
                if valid.any() else None)
        output['test'].append(row)
    output['test_status'] = 'compared' if len(common) else 'unavailable: no overlapping model/test dates'
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('baseline-log', 'candidate-log', 'baseline-model', 'candidate-model'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop('output')
    report = compare_runs(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False, default=lambda x: x.item()))


if __name__ == '__main__':
    main()
