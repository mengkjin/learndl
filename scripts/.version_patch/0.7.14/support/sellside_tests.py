"""Fresh tests for saved models affected by the Dongfang-to-Huayuan transition."""
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.feather as feather
import yaml

from src.automation.artifact_exchange.core import digest
from src.automation.console import logger
from src.automation.version_patch import load_patch_support
from src.proj import CALENDAR, MACHINE, PATH, Save
from src.res.factor.fmp.builder import FMP_RESUME_VERSION
from src.res.model.model_module.application import ModelTrainer
from src.shared.file_ops import atomic_json, fsync_dir

FACTORS = {f'df_scores_v{version}' for version in (0, 2, 3)}
FIRST = 20260101
PARTS = ('snapshot', 'results')
RECEIPT = 'sellside-model-tests.json'


def dates(base):
    records = base / 'snapshot/pred_recorder'
    saved = set()
    for path in (*records.glob('avg_preds/*.feather'), *records.glob('preds/*.feather')):
        saved.update(feather.read_table(path, columns=['date']).column('date').unique().to_pylist())
    missing = records / 'records/missing_pred_dates.json'
    skipped = json.loads(missing.read_text())['missing_dates'] if missing.is_file() else []
    return saved, set(skipped)


def candidates():
    result = [(PATH.model_factor / name, {name}) for name in sorted(FACTORS)]
    for path in sorted(PATH.model_linear.glob('*/configs/model.yaml')):
        config = yaml.safe_load(path.read_text())
        inputs = config.get('input.final.types') or []
        if isinstance(inputs, str):
            inputs = inputs.split(',')
        factors = {name for name in FACTORS if f'factor@{name}' in inputs}
        if factors:
            result.append((path.parents[1], factors))
    return result


def inventory(base):
    return {str(path.relative_to(base)): digest(path) for part in PARTS
            for path in sorted((base / part).rglob('*')) if path.is_file()}


def run_staged_test(base_path, **options):
    # The patch runner already holds exclusive maintenance and code leases.
    # A normal GO() acquires both again via record_training and would deadlock.
    from src.proj.util.memory_tracker import ProcessMemoryTracker
    from src.res.model.util import ModelPath

    trainer = ModelTrainer.initialize(base_path=ModelPath(base_path), **options)
    trainer.stage_setup()
    try:
        with ProcessMemoryTracker(f'sellside_replay.{base_path.name}') as tracker:
            trainer.memory_tracker = trainer.data.memory_tracker = tracker
            trainer.stage_data()
            with trainer.model_phase_log('test'):
                trainer.stage_test()
            trainer.stage_summary()
    finally:
        trainer.data.clear_disk_batches('test')
        for callback in trainer.callback.callbacks:
            if close := getattr(callback, 'reset_writer', None):
                close()
        Save.async_wait_all(raise_first_error=True, caller_name='sellside_model_tests')


def rebuild(base, staged, observed, skipped):
    shutil.copytree(base / 'configs', staged / 'configs')
    attr = f'model_{base.parent.name}'
    original = getattr(PATH, attr)
    # ModelPath resolves its output from these roots. This temporary tree is
    # inside PATH.model so model-path parsing still uses the native convention.
    setattr(PATH, attr, staged.parent)
    try:
        config = yaml.safe_load((base / 'configs/model.yaml').read_text())
        detail = dict(config.get('callbacks.DetailedAlphaAnalysis') or {})
        tasks = list(detail.get('tasks', ['factor', 't50']))
        for folder in sorted((base / 'snapshot/detailed_alpha').glob('*_fmp_test')):
            task = folder.name.removesuffix('_fmp_test')
            if task not in tasks:
                tasks.append(task)
        if (base / 'snapshot/detailed_alpha/factor_perf_test').is_dir() and 'factor' not in tasks:
            tasks.append('factor')
        detail['tasks'] = tasks
        options = {key: value for key, value in config.items() if key.startswith('input.final.')
                   or key == 'model.labels'}
        options['callbacks.DetailedAlphaAnalysis'] = detail
        run_staged_test(base_path=staged, override=options, stage=2, resume=0, selection=0, short_test=False,
                        start=config.get('model.beg_date'), end=config.get('model.end_date'), use_data='both')
    finally:
        setattr(PATH, attr, original)

    new, new_skipped = dates(staged)
    if not new or (observed | skipped) - (new | new_skipped) or not any(date >= FIRST for date in new):
        raise ValueError(f'{base.name}: incomplete prediction history after fresh test')
    # Detailed analysis callbacks may skip work. Check that every prior test
    # family still has its statistics, holdings and accounts before installation.
    analysis = Path('snapshot/detailed_alpha')
    groups = analysis / 'factor_perf_test/factor_stats/group_perf'
    if (base / groups).is_dir():
        from src.res.factor.analytic.factor_perf import FactorPerfTest
        from src.res.factor.util.classes.stock_factor import FactorStats

        for params in FactorPerfTest.required_group_stats():
            target = staged / groups / (FactorStats.param_to_str(params) + '.feather')
            if not target.is_file() or not target.stat().st_size:
                raise ValueError(f'{base.name}: missing rebuilt group statistics {target}')
    for path in (base / analysis).rglob('*'):
        if path.is_file() and path.suffix in ('.feather', '.tar'):
            if path.parent == base / groups:
                continue  # Daily resume uses the current grouping identities checked above.
            relative = path.relative_to(base / analysis)
            parts = relative.parts
            if len(parts) == 3 and parts[0].endswith('_fmp_test') and parts[1] in ('portfolio', 'account'):
                relative = Path(parts[0], FMP_RESUME_VERSION, *parts[1:])
            target = staged / analysis / relative
            if not target.is_file() or not target.stat().st_size:
                raise ValueError(f'{base.name}: missing rebuilt analysis {target}')
    for name in ('basic_test.xlsx', 'detailed_alpha_data.xlsx', 'detailed_alpha_plot.pdf'):
        if (base / 'results' / name).exists() and not (staged / 'results' / name).is_file():
            raise ValueError(f'{base.name}: missing rebuilt report {name}')
    if not (staged / 'snapshot/basic_test/test_summary.feather').is_file():
        raise ValueError(f'{base.name}: missing basic test summary')


def install(base, work, manifest):
    # The plan and staged directories survive a process kill. A retry finishes
    # outstanding renames without treating a half-installed model as an input.
    for part in PARTS:
        pending = work / 'new' / part
        backup = work / 'original' / part
        target = base / part
        if pending.exists():
            if target.exists():
                if backup.exists():
                    raise ValueError(f'Unexpected concurrent output {target}')
                target.rename(backup)
                fsync_dir(backup.parent)
                fsync_dir(base)
            pending.rename(target)
            fsync_dir(base)
            fsync_dir(pending.parent)
    if inventory(base) != manifest:
        raise ValueError(f'{base.name}: installed snapshot or report differs from staged files')


def apply(ctx):
    receipt = ctx.backup_dir / RECEIPT
    if not MACHINE.platform_server:
        atomic_json(receipt, {'receiver': True})
        return
    if ctx.root.resolve() != PATH.production.resolve():
        raise ValueError('Expected production checkout')
    old = load_patch_support('0.7.11', 'sellside_portfolios', root=ctx.root)
    outputs = {}
    for base, factors in candidates():
        relative = base.relative_to(PATH.model)
        work = PATH.model / '.version-patch-0.7.14-002' / relative
        plan = work / 'plan.json'
        if not plan.is_file():
            observed, skipped = dates(base)
            if not observed and not skipped and not any((base / 'snapshot').rglob('*.feather')):
                continue
            config = yaml.safe_load((base / 'configs/model.yaml').read_text())
            end = min(config.get('model.end_date') or 99991231, CALENDAR.updated())
            if end < FIRST:
                continue
            old._check_vendor_history({f'factor@{name}' for name in factors}, end)
            logger.info('Fresh test for %s through %s, affected factors %s', relative, end, sorted(factors))
            work.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=work) as directory:
                staged = Path(directory) / relative
                rebuild(base, staged, observed, skipped)
                manifest = inventory(staged)
                new = work / 'new'
                if new.exists():
                    shutil.rmtree(new)  # Uninstalled staging from before plan creation.
                staged.rename(new)
                (work / 'original').mkdir(exist_ok=True)
                atomic_json(plan, manifest)
        manifest = json.loads(plan.read_text())
        install(base, work, manifest)
        outputs[str(relative)] = manifest
    atomic_json(receipt, {'models': outputs})
    logger.info('Verified rebuilt model tests: %s', sorted(outputs))


def verify(ctx):
    receipt = ctx.backup_dir / RECEIPT
    if not receipt.is_file():
        return False
    state = json.loads(receipt.read_text())
    if not MACHINE.platform_server:
        return state.get('receiver') is True
    return 'models' in state and all(inventory(PATH.model / key) == value
                                    for key, value in state['models'].items())
