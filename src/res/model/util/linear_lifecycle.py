"""Linear test lifecycle; independent of null-model and checkpoint discovery rules."""
from __future__ import annotations

import shutil
from src.proj import PATH


def prepare_linear_run(config):
    from .config.config import get_config_dict
    from .final_source import FinalSource, final_weights
    weights = final_weights(config.input_final_weights, len(config.input_final_types))
    period = f'{config.beg_date}..{config.end_date}'
    for key, weight in zip(config.input_final_types, weights):
        if weight:
            FinalSource.parse(key).validate(period)
    base = config.base_path
    saved_path = base.conf_file('model')
    if config.is_resuming:
        if not saved_path.is_file():
            raise FileNotFoundError(f'Cannot resume linear test: missing {saved_path}')
        saved = get_config_dict(saved_path)
        # Period/extensions and refresh are operational; all other semantics must match.
        operational = {'model.beg_date', 'model.end_date', 'input.final.refresh'}
        keys = {key for key in set(saved.keys()) | set(config.Param.keys())
                if key.startswith(('input.', 'model.', 'train.criterion.')) and key not in operational}
        changed = sorted(key for key in keys if saved.get(key) != config.Param.get(key))
        if changed:
            raise ValueError(f'Linear configuration changed ({changed}); use resume=0 to recompute')
    else:
        # Only this schedule's generated outputs; never touch upstream models/caches.
        for path in (base.snapshot('pred_recorder'), base.snapshot('basic_test'),
                     base.snapshot('detailed_alpha'), base.rslt()):
            if path.is_dir():
                shutil.rmtree(path)
    PATH.model_linear.mkdir(parents=True, exist_ok=True)
