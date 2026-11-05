"""Compare complete T50 backends using stored real alpha/universe snapshots.

Does not train a model or rebuild risk/universe data. Each pair checks every
position and every rounded account field. Cold OS caches are NOT claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.proj import DB, Save
from src.res.factor.fmp.builder import PortfolioGroupBuilder
from src.res.factor.fmp.generator_array import run_t50
from src.res.factor.util import AlphaModel, Amodel, Portfolio
from src.res.factor.util.classes.universe import Universe


def load_inputs(alpha, universe, start, end):
    dates = np.asarray(DB.dates('model_prediction', alpha), dtype=int)
    dates = dates[(dates >= start) & (dates <= end)]
    if not len(dates):
        raise ValueError('no stored prediction dates in requested interval')
    models, frames, coverage = [], [], {}
    duplicate_rows = 0
    digest = hashlib.sha256()
    for date in dates:
        paths = [DB.path('model_prediction', alpha, int(date)), DB.path('universe', universe, int(date))]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f'benchmark requires stored inputs: {path}')
            digest.update(path.read_bytes())
        predictions = DB.load('model_prediction', alpha, int(date))
        duplicate_rows += int(predictions.duplicated('secid').sum())
        # DetailedAlphaAnalysis.get_factor uses pivot_table's default mean.
        predictions = predictions.groupby('secid', as_index=False)[alpha].mean()
        predictions = predictions.dropna(subset=[alpha])
        models.append(Amodel(int(date), predictions[alpha].to_numpy(float), predictions.secid.to_numpy(int), 'best'))
        frame = DB.load('universe', universe, int(date)).assign(date=int(date))
        for column in frame.columns:
            coverage[column] = coverage.get(column, 0) + 1
        frames.append(frame)
    # Use exactly the stored exclusion flags and the project's combination
    # semantics. Report coverage; do not silently rebuild missing warnst flags.
    universe_frame = Universe._combine_caches_with_exclusion(frames)
    portfolio = Portfolio.from_dataframe(universe_frame, name='univ')
    return AlphaModel('best', models), portfolio, dict(
        dates=len(dates), start=int(dates[0]), end=int(dates[-1]), input_sha256=digest.hexdigest(),
        stored_universe_column_date_counts=coverage, prediction_duplicate_rows_averaged=duplicate_rows,
        universe_note='Stored flags only; missing flags were not rebuilt. Both backends use identical inputs.')


def compare(left, right, left_frame, right_frame):
    pd.testing.assert_frame_equal(left_frame, right_frame, check_exact=True)
    for a, b in zip(left.builders, right.builders, strict=True):
        if list(a.portfolio.ports) != list(b.portfolio.ports):
            raise AssertionError('portfolio dates differ')
        for date, p in a.portfolio.ports.items():
            q = b.portfolio.ports[date]
            np.testing.assert_array_equal(p.secid, q.secid)
            np.testing.assert_allclose(p.weight, q.weight, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(p.value, q.value, rtol=1e-10, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha', default='gru_day_V1')
    parser.add_argument('--universe', default='top-1000')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trailing', type=int, default=20)
    parser.add_argument('--case', choices=('both', 'full', 'resume'), default='both')
    parser.add_argument('--first-backend', choices=('legacy', 'array'), default='legacy',
                        help='Run a separate fresh process with array first to measure its first-use cost too.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.trailing < 1:
        parser.error('repeats and trailing must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    alpha, universe, meta = load_inputs(args.alpha, args.universe, args.start, args.end)
    report: dict[str, Any] = dict(inputs=meta, input_loading_seconds=time.perf_counter() - started,
                  platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
                  source_sha256=hashlib.sha256(Path('src/res/factor/fmp/generator_array.py').read_bytes()).hexdigest(),
                  timing_note='First pair includes first-use process caches; later pairs are warm, alternating order. '
                              'Includes backend setup, quote preparation, materialization, save and async wait; '
                              'excludes common prediction/universe input loading, statistics and plotting.', runs=[])
    dates = np.sort(alpha.available_dates().dates)
    split = max(1, len(dates) - args.trailing)
    try:
        cases = ('full', 'resume') if args.case == 'both' else (args.case,)
        for repeat in range(args.repeats):
            order = ('legacy', 'array') if args.first_backend == 'legacy' else ('array', 'legacy')
            modes = order if repeat % 2 == 0 else order[::-1]
            for case in cases:
                outputs = {}
                for backend in modes:
                    path = args.output / f'{repeat}-{case}-{backend}'
                    def group(model):
                        return PortfolioGroupBuilder('top', model, [universe], analytic=False, attribution=False,
                            trade_engine='yale', n_best=50, start=args.start, end=args.end,
                            resume=True, resume_path=path, vb_level='never')
                    if case == 'resume':
                        # Identical legacy seed state, with one overlapping day
                        # to exercise trailing overwrite rather than append only.
                        initial = group(alpha.subset(dates[:split]))
                        run_t50(initial, 'legacy')
                        Save.async_wait_all(caller_name='t50_benchmark_seed')
                        model = alpha.subset(dates[max(0, split - 1):])
                    else:
                        model = alpha
                    target = group(model)
                    begin = time.perf_counter()
                    frame = run_t50(target, backend)
                    Save.async_wait_all(caller_name='t50_benchmark')
                    seconds = time.perf_counter() - begin
                    report['runs'].append(dict(repeat=repeat, case=case, backend=backend,
                                               seconds=seconds, diagnostics=getattr(target, 't50_diagnostics')))
                    outputs[backend] = (target, frame)
                    print(f'{repeat} {case} {backend}: {seconds:.3f}s', flush=True)
                compare(outputs['legacy'][0], outputs['array'][0], outputs['legacy'][1], outputs['array'][1])
                report['runs'][-1]['pair_equivalent'] = True
                (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        report['summary'] = {}
        for case in cases:
            samples = {mode: [r['seconds'] for r in report['runs'] if r['case'] == case and r['backend'] == mode]
                       for mode in ('legacy', 'array')}
            old, new = (statistics.median(samples[mode]) for mode in ('legacy', 'array'))
            report['summary'][case] = dict(legacy_median_seconds=old, array_median_seconds=new,
                                          speedup=old / new, reduction=1 - new / old)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
