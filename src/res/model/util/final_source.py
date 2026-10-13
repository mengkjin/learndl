"""Scalar result inputs: exact sources, rolling checkpoints and daily preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
import torch
from src.func.linear import linear_weights, weighted_components


def final_weights(value, size: int) -> np.ndarray:
    if not (isinstance(value, str) and value == 'equal') and not isinstance(value, (list, tuple)):
        raise ValueError('input.final.weights must be equal or a numeric vector')
    return linear_weights(value, size)


def prepare_final_values(values: torch.Tensor, precursor: str, weights) -> torch.Tensor:
    """Process [stock, date, source] on each common finite daily universe."""
    if values.ndim != 3:
        raise ValueError('Expected final values with stock, date, source axes')
    if precursor not in ('none', 'normal', 'percentile'):
        raise ValueError(f'Unknown final precursor: {precursor}')
    w = final_weights(weights, values.shape[-1])
    raw = values.detach().cpu().numpy().astype(float, copy=True)
    active = w != 0
    common = np.asarray(np.isfinite(raw[..., active]).all(axis=-1))
    result = np.full_like(raw, np.nan)
    for date in range(raw.shape[1]):
        mask = common[:, date]
        if not mask.any():
            continue
        x = raw[mask, date, :].copy()
        x[:, ~active] = 0
        if precursor == 'normal':
            mean = x.mean(axis=0)
            std = x.std(axis=0, ddof=0)
            x = np.divide(x - mean, std, out=np.zeros_like(x), where=std > 0)
            x = x.clip(-3.5, 3.5)
        elif precursor == 'percentile':
            x = pd.DataFrame(x).rank(method='average', pct=True).to_numpy()
        result[mask, date, :] = weighted_components(x, w)
    return torch.as_tensor(result, dtype=values.dtype, device=values.device)


@dataclass(frozen=True)
class FinalSource:
    kind: str
    reference: str
    column: str | None = None

    @property
    def expression(self):
        return f'{self.kind}@{self.reference}' + (f'@{self.column}' if self.column else '')

    def factor_calculator(self):
        from src.res.factor.calculator import FactorCalculator, StockFactorHierarchy
        if self.reference not in FactorCalculator.all_factors():
            raise ValueError(f'Unknown factor: {self.expression}')
        return StockFactorHierarchy.get_factor(self.reference)

    def storage_paths(self):
        """Use actual factor backing databases, including affiliate alternatives."""
        from src.proj import DB
        if self.kind == 'model_pred':
            from .model_source import model_source_path
            return [model_source_path(self.reference).base]
        if self.kind == 'factor':
            from src.res.factor.calculator import AffiliateFactorCalculator
            calc = self.factor_calculator()
            dbs = calc.full_dbs() if issubclass(calc, AffiliateFactorCalculator) else [
                {'src': calc.db_src, 'key': calc.db_key}]
        else:
            dbs = [{'src': 'pred', 'key': self.reference}]
        paths = []
        for db in dbs:
            path = DB.DBPath(str(db['src']), str(db['key']))
            paths.extend(p.parent if p.by_date else DB.DBPath.PathExact(p.src, p.key)
                         for p in [path, *path.alternatives()])
        return paths

    def validate(self, period=''):
        """Structural preflight, before a fresh run removes its previous results."""
        try:
            paths = self.storage_paths()
        except (FileNotFoundError, ValueError) as exc:
            raise type(exc)(f'{self.expression} {period}: {exc}') from exc
        if not any(path.exists() for path in paths):
            raise FileNotFoundError(f'{self.expression} {period}: no source storage exists: {paths}')
        return paths

    @classmethod
    def parse(cls, text: str) -> FinalSource:
        if not isinstance(text, str) or '@' not in text:
            raise ValueError(f'Invalid final source: {text!r}')
        kind, reference = text.split('@', 1)
        if kind not in ('pred', 'factor', 'model_pred') or not reference:
            raise ValueError(f'Invalid final source: {text!r}')
        column = None
        if kind == 'model_pred':
            parts = reference.rsplit('@', 1)
            if len(parts) == 2 and re.fullmatch(r'pred\.\d+', parts[1]):
                reference, column = parts
            from .model_source import parse_model_source
            parse_model_source(reference)  # syntax only; resolve files when loading
        elif not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*' if kind == 'pred' else r'[A-Za-z0-9_][A-Za-z0-9_.-]*', reference):
            raise ValueError(f'Invalid {kind} name: {reference}')
        return cls(kind, reference, column)

    def frame(self, dates, *, refresh=False) -> pd.DataFrame:
        """Return only requested dates; cache contents can include other dates."""
        if len(dates) == 0:
            return pd.DataFrame(columns=['secid', 'date', 'value'])
        if self.kind == 'model_pred':
            from src.res.model.model_module.application.predictor import ArchivedPredictorModel
            try:
                source = ArchivedPredictorModel.from_model_reference(self.reference)
            except (FileNotFoundError, ValueError) as exc:
                raise type(exc)(f'{self.expression} {min(dates)}..{max(dates)}: {exc}') from exc
            archives = source.model_dates
            indices = np.searchsorted(archives, dates, side='left') - 1
            # Earlier dates have no usable checkpoint; never use the first future model.
            frames = []
            for index in np.unique(indices[indices >= 0]):
                requested = np.asarray(dates)[indices == index]
                frame = source.pred_values(requested, int(archives[index]),
                                           load_first=not refresh, async_save=False)
                if frame.height:
                    frames.append(frame.to_pandas().query('date in @requested'))
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            if self.kind == 'pred':
                from src.data.loader import BlockLoader
                self.validate(f'{min(dates)}..{max(dates)}')
                block = BlockLoader('pred', self.reference).load(int(min(dates)), int(max(dates)))
                df = block.to_dataframe().reset_index() if not block.empty else pd.DataFrame()
            else:
                self.validate(f'{min(dates)}..{max(dates)}')
                df = self.factor_calculator().Loads(dates, normalize=False, closest=False)
        if df.empty:
            return pd.DataFrame(columns=['secid', 'date', 'value'])
        columns = [c for c in df.columns if c not in ('date', 'secid')]
        column = self.column
        if column is None:
            if len(columns) != 1:
                raise ValueError(f'{self.reference} has multiple output columns {columns}; select one explicitly')
            column = columns[0]
        if column not in columns:
            raise ValueError(f'{self.reference}: output column {column} does not exist')
        df = df.loc[df.date.isin(dates), ['secid', 'date', column]].rename(columns={column: 'value'})
        if df.duplicated(['secid', 'date']).any():
            raise ValueError(f'Duplicate stock/date rows in {self.reference}')
        return df


class FinalInputLoader:
    """Per-DataModule raw cache; preprocessing never mutates source caches."""
    def __init__(self, config):
        self.config = config
        self.sources = {key: FinalSource.parse(key) for key in config.input_final_types}
        self.frames: dict[str, pd.DataFrame] = {}
        self.loaded_dates: dict[str, set[int]] = {key: set() for key in self.sources}
        self.refreshed_dates: dict[str, set[int]] = {}
        self.coverage_by_date: dict[int, int] = {}
        self.source_dates: dict[str, set[int]] = {key: set() for key in self.sources}

    def validate_coverage(self, dates):
        """Check the whole attempted test period, never an individual empty batch."""
        attempted = sorted(set(map(int, dates)) & self.coverage_by_date.keys())
        if not attempted:
            return 'Final inputs: no new dates to evaluate'
        period = f'{attempted[0]}..{attempted[-1]}'
        weights = final_weights(self.config.input_final_weights, len(self.sources))
        for (key, source), weight in zip(self.sources.items(), weights):
            if weight and not self.source_dates[key].intersection(attempted):
                raise ValueError(f'{key}: no finite values during {period}; storage: {source.storage_paths()}')
        skipped = [date for date in attempted if not self.coverage_by_date[date]]
        if len(skipped) == len(attempted):
            raise ValueError(f'Final sources {list(self.sources)}: no common finite stocks during {period}')
        return (f'Final coverage {period}: {len(attempted) - len(skipped)}/{len(attempted)} usable dates; '
                f'skipped dates: {skipped}')

    def blocks(self, dates, secid, align_date):
        from src.data.util import DataBlock
        dates = np.unique(dates)
        weights = final_weights(self.config.input_final_weights, len(self.sources))
        blocks = {}
        for i, (key, source) in enumerate(self.sources.items()):
            pending = np.array(sorted(set(dates) - self.loaded_dates[key]), dtype=int)
            if len(pending) and weights[i] != 0:
                if source.kind == 'model_pred' and self.config.input_final_refresh:
                    refreshed = self.refreshed_dates.setdefault(source.reference, set())
                    fresh = np.array([d for d in pending if d not in refreshed], dtype=int)
                    cached = np.array([d for d in pending if d in refreshed], dtype=int)
                    pieces = []
                    if len(fresh):
                        pieces.append(source.frame(fresh, refresh=True))
                        refreshed.update(fresh)
                    if len(cached):
                        pieces.append(source.frame(cached, refresh=False))
                    frame = pd.concat(pieces, ignore_index=True)
                else:
                    frame = source.frame(pending)
                old = self.frames.get(key)
                self.frames[key] = pd.concat([old, frame], ignore_index=True) if old is not None else frame
            self.loaded_dates[key].update(pending)
            frame = self.frames.get(key, pd.DataFrame(columns=['secid', 'date', 'value']))
            frame = frame.loc[frame.date.isin(dates)]
            # Explicit grids preserve missing days/stocks even for wholly empty sources.
            index = pd.MultiIndex.from_product([list(secid), dates.tolist()], names=['secid', 'date'])
            panel = frame.set_index(['secid', 'date'])[['value']].reindex(index).astype(float)
            blocks[key] = DataBlock.from_pandas(panel)
        values = torch.stack([b.values[:, :, 0, 0] for b in blocks.values()], dim=-1)
        for i, key in enumerate(self.sources):
            finite_dates = torch.isfinite(values[..., i]).any(dim=0).cpu().numpy()
            self.source_dates[key].update(map(int, dates[finite_dates]))
        values = prepare_final_values(values, self.config.input_final_precursor, self.config.input_final_weights)
        self.coverage = torch.isfinite(values).all(dim=-1).sum(dim=0).tolist()
        self.coverage_by_date.update(zip(map(int, dates), self.coverage))
        for i, (key, block) in enumerate(blocks.items()):
            block.update(values=values[..., i, None, None], feature=[key], inplace=True)
            blocks[key] = block.align_secid_date(secid, align_date)
        return blocks
