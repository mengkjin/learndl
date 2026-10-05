"""Sampled cross-sectional comparisons and exact historical checkpoint routing."""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from .statistics import correlation, linear_cka, sample_dates


def checkpoint_for_date(archive_dates, date):
    eligible = [int(d) for d in archive_dates if int(d) < date]
    if not eligible:
        raise ValueError(f'No checkpoint strictly before {date}')
    return max(eligible)


class ArchivedOutputProvider:
    """Read training records without invoking PredRecorder's mutating getters.

    Newly inferred representations use the existing per-checkpoint caches. The
    exact-source predictor handles device selection and sequence warm-up.
    """
    def __init__(self, model, config):
        self.model, self.config = model, config
        self.root = model.inference_dir
        self.sources = {}
        self.frames = {}
        self.dates_by_num = {}
        if self.root is None:
            raise ValueError('No inference_dir associated with this Excel report')
        for folder in sorted((self.root / 'archive').glob('*')):
            if folder.is_dir() and folder.name.isdigit():
                self.dates_by_num[int(folder.name)] = sorted(
                    int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit())
        # Prediction snapshots remain usable after archives have been removed.
        for path in (self.root / 'snapshot/pred_recorder/preds').glob('*.feather'):
            parts = path.stem.split('.')
            if len(parts) == 4 and all(p.isdigit() for p in parts):
                num, checkpoint = map(int, parts[:2])
                self.dates_by_num.setdefault(num, [])
                self.dates_by_num[num] = sorted(set(self.dates_by_num[num]) | {checkpoint})
        self.avg_paths = sorted((self.root / 'snapshot/pred_recorder/avg_preds').glob('*.feather'))
        if not self.dates_by_num and self.avg_paths:
            self.dates_by_num[0] = sorted({int(p.stem.split('.')[0]) for p in self.avg_paths})
        if not self.dates_by_num:
            raise ValueError('No archived model numbers or training prediction records')

    def _read(self, path):
        if path not in self.frames:
            self.frames[path] = pd.read_feather(path) if path.is_file() else pd.DataFrame()
        return self.frames[path]

    def available_dates(self, kind, dates):
        nums = sorted(self.dates_by_num) if kind == 'pred' else [self.model.hidden_model_num]
        return [d for d in dates if all(any(c < d for c in self.dates_by_num.get(n, [])) for n in nums)]

    def _source(self, num):
        if num not in self.sources:
            from src.res.model.util import ModelPath
            from src.res.model.model_module.application.predictor import ArchivedPredictorModel
            path = ModelPath(self.root)
            reference = f'{path.full_name}@{num}@{self.config.submodel}'
            # full_name may omit @1; the exact source parser accepts that form.
            self.sources[num] = ArchivedPredictorModel.from_model_str(reference)
        return self.sources[num]

    def _recorded(self, num, checkpoint, date):
        folder = self.root / 'snapshot/pred_recorder/preds'
        frames = []
        for path in sorted(folder.glob(f'{num}.{checkpoint}.*.feather')):
            parts = path.stem.split('.')
            if len(parts) == 4 and int(parts[2]) <= date <= int(parts[3]):
                df = self._read(path)
                sub = df.loc[(df.date == date) & (df.submodel == self.config.submodel)]
                if len(sub):
                    frames.append(sub)
        if not frames:
            return None
        df = pd.concat(frames)
        if df.duplicated(['secid', 'date']).any():
            raise ValueError('Duplicate training predictions for the selected checkpoint/date')
        return df.set_index('secid').pred

    def _values(self, num, checkpoint, date, kinds):
        paths = {k: self.root / 'snapshot' / f'{k}_values' /
                 f'{num}.{checkpoint}.{self.config.submodel}.feather' for k in kinds}
        values, missing = {}, []
        for kind, path in paths.items():
            df = self._read(path)
            if not df.empty and {'secid', 'date'}.issubset(df.columns):
                selected = df.loc[df.date == date]
                columns = [c for c in df if c.startswith(f'{kind}.')]
                if len(selected) and columns:
                    if selected.secid.duplicated().any():
                        raise ValueError(f'Duplicate {kind} cache keys')
                    values[kind] = selected.set_index('secid')[columns]
                    continue
            missing.append(kind)
        if missing:
            source = self._source(num)
            try:
                for batch in source.iter_batch_data([date], checkpoint, model_num=num,
                                                   submodel=self.config.submodel, require_grad=False):
                    if batch.batch_date != date or batch.output.empty or date in source.data.early_test_dates:
                        continue
                    for kind in missing:
                        if kind == 'hidden' and 'hidden' not in batch.output.other:
                            continue
                        df = batch.hidden_df() if kind == 'hidden' else batch.pred_df(label=False)
                        df = df.loc[df.date == date].copy()
                        if df.empty:
                            continue
                        if df.duplicated(['secid', 'date']).any():
                            raise ValueError(f'Duplicate inferred {kind} keys')
                        old = self._read(paths[kind])
                        saved = pd.concat([old, df], ignore_index=True).drop_duplicates(['secid', 'date'], keep='last')
                        from src.proj import Save
                        Save.df(saved, paths[kind], overwrite=True, async_save=False, vb_level='never')
                        self.frames[paths[kind]] = saved
                        values[kind] = df.set_index('secid').filter(regex=f'^{kind}\\.')
            finally:
                source.data.storage.del_group('retrospective')
        return values

    def get(self, date, want_pred, want_hidden):
        averaged = None
        if want_pred:
            checkpoints_at_date = {checkpoint_for_date(ds, date) for ds in self.dates_by_num.values()}
            if len(checkpoints_at_date) == 1:
                checkpoint = next(iter(checkpoints_at_date))
                matches = []
                for path in self.avg_paths:
                    parts = path.stem.split('.')
                    if len(parts) == 3 and int(parts[0]) == checkpoint and int(parts[1]) <= date <= int(parts[2]):
                        df = self._read(path)
                        matches.append(df.loc[(df.date == date) & (df.submodel == self.config.submodel)])
                if matches:
                    df = pd.concat(matches)
                    if df.secid.duplicated().any():
                        raise ValueError('Duplicate averaged training predictions')
                    if not df.empty:
                        averaged = df.set_index('secid').pred
        nums = sorted(self.dates_by_num) if want_pred else []
        if want_hidden:
            nums = sorted(set(nums) | {self.model.hidden_model_num})
        pred, hidden, checkpoints, errors = [], None, {}, {}
        for num in nums:
            score = None
            try:
                checkpoint = checkpoint_for_date(self.dates_by_num.get(num, []), date)
                checkpoints[num] = checkpoint
                score = averaged if averaged is not None else (self._recorded(num, checkpoint, date) if want_pred else None)
                need_hidden = want_hidden and num == self.model.hidden_model_num
                kinds = (['pred'] if want_pred and score is None else []) + (['hidden'] if need_hidden else [])
                values = self._values(num, checkpoint, date, kinds) if kinds else {}
                if need_hidden:
                    hidden = values.get('hidden')
                    if hidden is None:
                        errors['hidden'] = 'Model has no hidden output or no valid batch'
                if want_pred and score is None:
                    output = values.get('pred')
                    if output is None:
                        raise ValueError('No prediction output')
                    which = self._source(num).config.model_param[num].get('which_output', 0)
                    score = output.mean(axis=1) if which is None else output[f'pred.{which}']
                if want_pred:
                    if score is None:
                        raise ValueError('No prediction score')
                    pred.append(score.rename(num))
            except (FileNotFoundError, ValueError, KeyError, AssertionError, RuntimeError, IndexError) as exc:
                if want_pred:
                    if score is not None:
                        pred.append(score.rename(num))
                    else:
                        errors['pred'] = f'model_num={num}: {exc}'
                if want_hidden and num == self.model.hidden_model_num:
                    errors['hidden'] = f'model_num={num}: {exc}'
        score = None
        if want_pred and 'pred' not in errors and pred:
            score = pd.concat(pred, axis=1).mean(axis=1).rename('pred')
        return score, hidden, checkpoints, errors


def analyze_outputs(inputs, config, candidate_dates, provider_factory=ArchivedOutputProvider):
    """Return tables, matrices, sampling audit and diagnostics; no report rendering."""
    diagnostics, audit, tables, matrices = [], [], {}, {}
    providers = {}
    for model in inputs:
        try:
            providers[model.name] = provider_factory(model, config)
        except (ValueError, FileNotFoundError) as exc:
            diagnostics.append({'template': 'outputs', 'model': model.name, 'reason': str(exc)})
    selections = {}
    for kind, enabled, limit in [('pred', config.analyze_pred, config.sample_num_corr),
                                  ('hidden', config.analyze_hidden, config.sample_num_hidden)]:
        if not enabled:
            continue
        eligible = {}
        for name, provider in providers.items():
            dates = provider.available_dates(kind, candidate_dates)
            if dates:
                eligible[name] = set(dates)
            else:
                diagnostics.append({'template': kind, 'model': name, 'reason': 'No eligible checkpoint dates'})
        common = sorted(set.intersection(*eligible.values())) if len(eligible) >= 2 else []
        chosen = sample_dates(common, limit)
        selections[kind] = (chosen, eligible)
        audit.append({'kind': kind, 'candidate_count': len(common), 'limit': limit,
                      'selected_count': len(chosen), 'date': None, 'model': '*',
                      'status': 'selected' if chosen else 'unavailable'})
        if not chosen:
            diagnostics.append({'template': kind, 'reason': 'Fewer than two eligible models or no common dates'})
    all_dates = sorted(set(d for dates, _ in selections.values() for d in dates))
    pred_rows, cka_rows, dimension_frames = [], [], []
    for date in all_dates:
        preds, hiddens, checkpoint_labels = {}, {}, {}
        for name, provider in providers.items():
            want = {k: date in dates and name in eligible for k, (dates, eligible) in selections.items()}
            if not any(want.values()):
                continue
            try:
                score, hidden, checkpoints, errors = provider.get(date, want.get('pred', False), want.get('hidden', False))
            except (ValueError, FileNotFoundError, KeyError, RuntimeError, AssertionError) as exc:
                score, hidden, checkpoints = None, None, {}
                errors = {kind: str(exc) for kind in want}
            checkpoint_labels[name] = ';'.join(f'{n}:{d}' for n, d in sorted(checkpoints.items()))
            for kind, value in [('pred', score), ('hidden', hidden)]:
                if not want.get(kind):
                    continue
                success = value is not None and not value.empty
                audit.append({'kind': kind, 'date': date, 'model': name,
                              'status': 'ok' if success else 'failed',
                              'checkpoint': checkpoint_labels[name], 'n_stocks': len(value) if value is not None else 0,
                              'reason': errors.get(kind, '' if success else 'No output')})
                if success:
                    (preds if kind == 'pred' else hiddens)[name] = value
                else:
                    diagnostics.append({'template': kind, 'date': date, 'model': name,
                                        'reason': errors.get(kind, 'No output')})
        if len(preds) >= 2:
            for a, b in combinations_with_replacement(preds, 2):
                pair = pd.concat([preds[a].rename('a'), preds[b].rename('b')], axis=1)
                for method in ('pearson', 'spearman'):
                    value, n, reason = correlation(pair.a, pair.b, method)
                    pred_rows.append(dict(date=date, model_a=a, model_b=b, method=method, value=value, n=n, reason=reason))
        if len(hiddens) >= 2:
            for a, b in combinations_with_replacement(hiddens, 2):
                common = hiddens[a].index.intersection(hiddens[b].index)
                x, y = hiddens[a].loc[common], hiddens[b].loc[common]
                value, n, reason = linear_cka(x, y)
                cka_rows.append(dict(date=date, model_a=a, model_b=b, value=value, n=n, reason=reason))
                # Cython pairwise correlation avoids Python loops over every stock/feature pair.
                xp, yp = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
                joint = pd.DataFrame(np.c_[xp, yp]).replace([np.inf, -np.inf], np.nan)
                cross = joint.corr(min_periods=3).iloc[:x.shape[1], x.shape[1]:].to_numpy()
                counts = (np.isfinite(xp).astype(float).T @ np.isfinite(yp).astype(float)).astype(int)
                reasons = np.where(counts < 3, 'fewer than 3 paired observations',
                                   np.where(np.isnan(cross), 'constant values', ''))
                dimension_frames.append(pd.DataFrame({
                    'feature_a': np.repeat(x.columns.to_numpy(), y.shape[1]),
                    'feature_b': np.tile(y.columns.to_numpy(), x.shape[1]),
                    'value': cross.ravel(), 'n': counts.ravel(), 'reason': reasons.ravel(),
                }).assign(date=date, model_a=a, model_b=b,
                          checkpoint_a=checkpoint_labels[a], checkpoint_b=checkpoint_labels[b]))
        for kind, values in [('pred', preds), ('hidden', hiddens)]:
            if kind in selections and date in selections[kind][0] and len(values) < 2:
                diagnostics.append({'template': kind, 'date': date, 'reason': 'Fewer than two successful outputs'})
    names = [m.name for m in inputs]
    for key, rows in [('pred_daily', pred_rows), ('hidden_cka_daily', cka_rows)]:
        df = pd.DataFrame(rows)
        tables[key] = df
        if df.empty:
            continue
        groups = ['model_a', 'model_b'] + (['method'] if key == 'pred_daily' else [])
        summary = df.groupby(groups).value.agg(['mean', 'std', 'count']).reset_index()
        tables[key.replace('daily', 'summary')] = summary
        for method in (['pearson', 'spearman'] if key == 'pred_daily' else ['cka']):
            sub = summary.query('method == @method') if 'method' in summary else summary
            matrix = pd.DataFrame(np.nan, index=names, columns=names)
            counts = matrix.copy()
            for _, row in sub.iterrows():
                matrix.loc[row.model_a, row.model_b] = matrix.loc[row.model_b, row.model_a] = row['mean']
                counts.loc[row.model_a, row.model_b] = counts.loc[row.model_b, row.model_a] = row['count']
            prefix = f'pred_{method}' if key == 'pred_daily' else 'hidden_cka'
            matrices[f'{prefix}_mean'], matrices[f'{prefix}_count'] = matrix, counts
    tables['hidden_dimensions'] = pd.concat(dimension_frames, ignore_index=True) if dimension_frames else pd.DataFrame()
    audit_df = pd.DataFrame(audit)
    if not audit_df.empty:
        for kind in selections:
            success = audit_df[(audit_df.kind == kind) & (audit_df.status == 'ok')]
            completed = success.groupby('date').model.nunique()
            mask = (audit_df.kind == kind) & (audit_df.model == '*')
            audit_df.loc[mask, 'successful_dates'] = int((completed >= 2).sum())
    return tables, matrices, audit_df, diagnostics
