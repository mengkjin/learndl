"""Interactive model comparison from the Research Operations menu."""
from __future__ import annotations

from dataclasses import asdict
import json

from src.api.util.direct_call import DirectCall
from src.proj import Logger, Options, PATH
from src.proj.util.cli import AskFor

__all__ = ['CompareModels']


def comparison_schema():
    """Reuse the standard CLI defaults/customize flow with comparison defaults."""
    from src.api.util.backend import ScriptParamInput
    from src.proj.util.script.param_schema import ScriptParamSchema
    from src.res.model.analytic.compare import CompareConfig

    defaults = asdict(CompareConfig())
    defaults.pop('trade_dates')
    defaults['periods'] = ','.join(defaults['periods'])
    defaults['templates'] = ','.join(defaults['templates'])
    defaults['custom_periods'] = '{}'
    defaults.update(hidden_model_num=0, output_dir=None)
    descriptions = {
        'start': 'Start date YYYYMMDD; None uses common coverage',
        'end': 'End date YYYYMMDD; None uses common coverage',
        'periods': 'Periods, comma-separated: all,year,recent_year,quarter,month',
        'templates': 'Templates, comma-separated: ic,top,complementarity',
        'custom_periods': 'Named periods as JSON, e.g. {"holdout": [20250101, 20251231]}',
        'rolling_window': 'IC rolling window (number of observations)',
        'analyze_pred': 'Analyze sampled prediction correlations (may require inference)',
        'analyze_hidden': 'Analyze sampled hidden representations (may require inference)',
        'sample_num_corr': 'Maximum prediction sample dates',
        'sample_num_hidden': 'Maximum hidden sample dates',
        'hidden_model_num': 'Hidden replica number for each selected model',
        'output_dir': 'Output directory; None creates results/model_compare/<timestamp>/',
    }
    params = []
    for name, value in defaults.items():
        kind = 'int' if name in ('start', 'end') else type(value).__name__
        if value is None and name not in ('start', 'end'):
            kind = 'str'
        params.append(ScriptParamInput(
            name=name, type=['pearson', 'spearman'] if name == 'corr_method' else kind,
            desc=descriptions.get(name, name.replace('_', ' ').title()), default=value,
        ))
    return ScriptParamSchema(PATH.scpt / '5_test/2_compare_models.py', params, defaults)


def parse_parameters(values):
    from src.res.model.analytic.compare import CompareConfig

    values = dict(values)
    replica = values.pop('hidden_model_num')
    output_dir = values.pop('output_dir')
    if not isinstance(replica, int) or isinstance(replica, bool) or replica < 0:
        raise ValueError('hidden_model_num must be a non-negative integer')
    for name in ('periods', 'templates'):
        values[name] = tuple(part.strip() for part in values[name].split(',') if part.strip())
    values['custom_periods'] = json.loads(values['custom_periods'])
    if not isinstance(values['custom_periods'], dict):
        raise ValueError('custom_periods must be a JSON object')
    config = CompareConfig(**values)
    config.validate()
    for name in ('submodel', 'ic_benchmark', 'top_sheet', 'top_strategy', 'top_benchmark', 'top_suffix'):
        if not getattr(config, name):
            raise ValueError(f'{name} cannot be empty')
    return config, replica, output_dir


class CompareModels(DirectCall):
    """Select models, configure comparison, and export Excel/PDF reports."""
    category = 'Research'

    def run(self) -> None:
        choices = [name for name in Options.available_models(refresh=True)
                   if (PATH.model / name / 'results/detailed_alpha_data.xlsx').is_file()]
        if len(choices) < 2:
            Logger.warning('Model comparison needs at least two models with results/detailed_alpha_data.xlsx.')
            return
        while True:
            selected = AskFor.Selections(
                choices, multiple=True, confirm=False, use_checkbox=True,
                title='Select at least two models to compare',
                help_description='Only models with results/detailed_alpha_data.xlsx are listed.',
            )
            if not selected.valid:
                return
            if len(selected.results) >= 2:
                break
            Logger.warning('Select at least two models to compare.')

        Logger.note('Selected models: ' + ', '.join(choices[i - 1] for i in selected.results))
        schema = comparison_schema()
        while True:
            parameters = AskFor.ScriptKwargs(
                schema, help_description='Use comparison defaults or customize dates, metrics and output analysis.',
            )
            if not parameters.valid or parameters.result is None:
                return
            try:
                config, replica, output_dir = parse_parameters(parameters.result)
            except (TypeError, ValueError, AttributeError) as exc:
                Logger.warning(f'Invalid comparison parameters: {exc}')
                continue
            break

        from src.res.model.analytic.compare import CompareModelSpec, compare_models
        models = [CompareModelSpec(PATH.model / choices[i - 1], hidden_model_num=replica)
                  for i in selected.results]
        Logger.note('Calculating model comparison and exporting Excel/PDF...')
        result = compare_models(models, config=config, output_dir=output_dir, display=True)
        for kind, path in result.output_paths.items():
            Logger.note(f'{kind}: {path}')
