"""Resolve a checked, frozen server dataset when an ordinary schedule starts."""
import json
import re


def resolve_schedule(source, model, name):
    from src.proj import PATH, CALENDAR
    from .bundle import load
    from .schedule_training import schedule_config, variants
    from pathlib import Path
    if not re.fullmatch(r'[A-Za-z0-9_-]+', source):
        raise ValueError('Invalid factor sequence source suite')
    if model not in dict(variants(False)):
        raise ValueError('Unsupported factor sequence model')
    suite = PATH.result / 'factor_sequence' / source
    report = json.loads((suite / 'suite.json').read_text())
    dataset = Path(report['dataset'])
    manifest = load(dataset)
    audit = json.loads((dataset / 'audit.json').read_text())
    if (audit.get('status') != 'passed' or audit.get('version') != manifest['version']
            or report.get('data_version') != manifest['version']):
        raise ValueError('Source suite requires a matching passed dataset audit')
    plan = json.loads((suite / 'walk_forward.json').read_text())
    folds = [{**f, 'model_date': int(CALENDAR.td(f['test'][0], -1))} for f in plan['folds']]
    if not folds:
        raise ValueError('Source suite has no historical folds')
    return schedule_config(dataset, name, model, folds, research=False, seed=17)
