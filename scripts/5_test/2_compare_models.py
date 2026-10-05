#!/usr/bin/env python
"""Compare at least two training directories or exported Excel reports.

Examples:
    python scripts/5_test/2_compare_models.py a.xlsx b.xlsx --start 20200101
    python scripts/5_test/2_compare_models.py --pred --hidden --sample-num-hidden 10
    python scripts/5_test/2_compare_models.py --specs comparison.json

The JSON file is a list of CompareModelSpec dictionaries and permits explicit
Excel-to-archive associations and per-model hidden_model_num choices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='*')
    parser.add_argument('--specs', type=Path)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--periods', nargs='+', default=['all', 'year', 'recent_year'])
    parser.add_argument('--custom-period', action='append', default=[], metavar='NAME:START:END')
    parser.add_argument('--templates', nargs='+', default=['ic', 'top', 'complementarity'])
    parser.add_argument('--submodel', default='best')
    parser.add_argument('--ic-benchmark', default='market')
    parser.add_argument('--top-sheet', default='t50@perf_curve')
    parser.add_argument('--top-strategy', default='Top_50')
    parser.add_argument('--top-benchmark', default='univ')
    parser.add_argument('--top-suffix', default='lag0')
    parser.add_argument('--top-n', type=int, default=50)
    parser.add_argument('--rolling-window', type=int, default=20)
    parser.add_argument('--corr-method', choices=['pearson', 'spearman'], default='pearson')
    parser.add_argument('--pred', action='store_true')
    parser.add_argument('--hidden', action='store_true')
    parser.add_argument('--sample-num-corr', type=int, default=20)
    parser.add_argument('--sample-num-hidden', type=int, default=20)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-display', action='store_true')
    args = parser.parse_args(argv)
    from src.res.model.analytic.compare import CompareConfig, CompareModelSpec, compare_models
    models = list(args.models)
    if args.specs:
        models.extend(CompareModelSpec(**s) for s in json.loads(args.specs.read_text()))
    if not models:
        from src.proj import PATH, Options
        from src.proj.util.cli import AskFor
        choices = [name for name in Options.available_models()
                   if (PATH.model / name / 'results/detailed_alpha_data.xlsx').is_file()]
        selection = AskFor.Selections(choices, multiple=True, confirm=False, use_checkbox=True,
                                      title='Select at least two models to compare')
        if not selection:
            return None
        models = [PATH.model / choices[i - 1] for i in selection.results]
    if len(models) < 2:
        parser.error('Select at least two models')
    custom = {}
    for value in args.custom_period:
        name, start, end = value.split(':')
        custom[name] = (int(start), int(end))
    kwargs = {k: getattr(args, k) for k in (
        'start', 'end', 'periods', 'templates', 'submodel', 'ic_benchmark', 'top_sheet',
        'top_strategy', 'top_benchmark', 'top_suffix', 'top_n', 'rolling_window',
        'corr_method', 'sample_num_corr', 'sample_num_hidden')}
    config = CompareConfig(**kwargs, custom_periods=custom, analyze_pred=args.pred, analyze_hidden=args.hidden)
    result = compare_models(models, config=config, output_dir=args.output_dir, display=not args.no_display)
    for kind, path in result.output_paths.items():
        print(f'{kind}: {path}')
    return result


if __name__ == '__main__':
    main()
