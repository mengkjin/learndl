# coding: utf-8
# author: auto
# date: 2026-08-06
# description: Recalc CNE5 residual downstream data (non-NN)
# content: |
#   After TuShareCNE5 calc_model residual-universe fix, rebuild affected
#   non-training-model artifacts in dependency order:
#     0) trade_ts/adjprice (halt-filled adjusted OHLC; required by labels)
#     1) CNE5 exp/coef/res then cov/spec (forced [start,end]; ignores testing schedule)
#     2) ClassicLabelsUpdater (res_lag* columns)
#     3) Stock factors that call RISK.get_exret
#     4) Preprocess dumps: y + style (fit & predict)
#     5) Purge ModuleData DataCache
#   Does NOT retrain NN/boost models or touch raw trade_ts day downloads.
# email: True
# mode: shell
# parameters:
#   start:
#       type: int
#       desc: inclusive start yyyyMMdd (CNE5 START_DATE default)
#       required: False
#       default: 20050101
#   end:
#       type: int
#       desc: inclusive end yyyyMMdd; 0 = calendar updated
#       required: False
#       default: 0
#   do_adjprice:
#       type: [True, False]
#       desc: backfill trade_ts/adjprice (halt-filled adjusted prices)
#       required: False
#       default: True
#   do_cne5:
#       type: [True, False]
#       desc: force CNE5 exp/coef/res/cov/spec over [start,end] (no testing schedule clamp)
#       required: False
#       default: True
#   do_labels:
#       type: [True, False]
#       desc: overwrite ClassicLabelsUpdater (labels_ts)
#       required: False
#       default: True
#   do_factors:
#       type: [True, False]
#       desc: fix stock factors that use get_exret
#       required: False
#       default: True
#   do_preprocess:
#       type: [True, False]
#       desc: reconstruct preprocess dumps y+style (fit+predict)
#       required: False
#       default: True
#   do_cache:
#       type: [True, False]
#       desc: purge ModuleData DataCache under PATH.datacache
#       required: False
#       default: True
#   dry_run:
#       type: [True, False]
#       desc: only print planned steps without writing
#       required: False
#       default: False

from __future__ import annotations

from src.proj import CALENDAR , Dates , Logger , PATH , Base
from src.proj.util.script import ScriptTool
from src.api.util.wrapper import wrap_update

# Factors that read RISK.get_exret (residual returns).
EXRET_FACTORS : tuple[str , ...] = (
    'exret_std1m' , 'exret_std2m' , 'exret_std3m' , 'exret_std6m' , 'exret_std12m' ,
    'mom_aog' , 'mom_aaa' ,
)

# Preprocess keys whose dumps embed CNE5 exp and/or labels_ts residuals.
PREPROCESS_KEYS : tuple[str , ...] = ('y' , 'style')


def _resolve_end(end: int) -> int:
    return int(end) if int(end) > 0 else int(CALENDAR.updated())


def _force_recalc_cne5(start: int , end: int , **kwargs) -> Base.UpdateFlag:
    """
    Force CNE5 ``exp/coef/res`` then ``cov/spec`` over ``[start, end]``.

    Bypasses ``TuShareCNE5_Calculator.recalculate`` → ``CALENDAR.update_schedule``,
    which on a testing machine schedule collapses the range to a few recent days
    (e.g. 20250416–20250422) and leaves historical ``tushare_cne5_res`` untouched.
    """
    from src.res.factor.risk import TuShareCNE5_Calculator

    start = max(int(start) , int(TuShareCNE5_Calculator.START_DATE))
    end = min(int(end) , int(CALENDAR.updated()))
    assert start <= end , f'empty CNE5 range after clamp: start={start} end={end}'

    # Exact trading-day span — do NOT call CALENDAR.update_schedule.
    dates = Dates(CALENDAR.range(start , end , 'td'))
    Logger.note(
        f'Force CNE5 update_date over {dates} '
        f'(bypass testing update_schedule; n={len(dates)})' ,
    )
    if dates.empty:
        return Base.UpdateFlag.SKIPPED

    updater = TuShareCNE5_Calculator()
    flags = Base.UpdateFlagList()
    for date in dates:
        flags += updater.update_date(int(date) , 'exposure')
    for date in dates:
        flags += updater.update_date(int(date) , 'risk')
    return flags.summarize()


@ScriptTool('recalc_cne5_downstream')
def main(
    start: int = 20050101 ,
    end: int = 0 ,
    do_adjprice: bool = True ,
    do_cne5: bool = True ,
    do_labels: bool = True ,
    do_factors: bool = True ,
    do_preprocess: bool = True ,
    do_cache: bool = True ,
    dry_run: bool = False ,
    **kwargs ,
):
    """
    Recalculate CNE5 residual-dependent data layers (not NN/boost checkpoints).

    Dependency order
    ----------------
    ``trade_ts/adjprice``
      → CNE5 ``exp/coef/res`` → ``cov/spec``
      → ``labels_ts`` (res_lag*)
      → stock factors using ``get_exret``
      → preprocess dumps ``y`` / ``style``
      → ModuleData disk cache
    """
    start = int(start)
    end = _resolve_end(end)
    dry_run = bool(dry_run)

    Logger.note(f'CNE5 downstream recalc  start={start} end={end} dry_run={dry_run}')
    Logger.stdout(
        'Affected chain: adjprice → cne5(exp/coef/res/cov/spec) → labels_ts → '
        f'factors{list(EXRET_FACTORS)} → preprocess{list(PREPROCESS_KEYS)} → datacache' ,
        indent = 1 ,
    )
    Logger.stdout(
        'NOT in scope: raw trade_ts/day download, day/30m/dfl2 preprocess, NN/boost training, '
        'indus preprocess (industry dummies unchanged by resid fix)' ,
        indent = 1 ,
    )

    steps = [
        ('adjprice' , do_adjprice) ,
        ('cne5' , do_cne5) ,
        ('labels' , do_labels) ,
        ('factors' , do_factors) ,
        ('preprocess' , do_preprocess) ,
        ('cache' , do_cache) ,
    ]
    Logger.stdout('Enabled steps: ' + ', '.join(name for name , on in steps if on) , indent = 1)

    if dry_run:
        Logger.warning('dry_run=True — no writes')
        return

    if do_adjprice:
        from src.data.update.custom.adjprice import AdjPriceUpdater
        # Chronological overwrite so halt forward-fill can read previous day.
        wrap_update(
            AdjPriceUpdater.proceed_update ,
            'overwrite trade_ts/adjprice (halt-filled adjusted OHLC/VWAP)' ,
            start = start ,
            end = end ,
            overwrite = True ,
        )

    if do_cne5:
        wrap_update(
            _force_recalc_cne5 ,
            'force CNE5 exp/coef/res then cov/spec over [start,end]' ,
            start = start ,
            end = end ,
        )

    if do_labels:
        from src.data.update.custom.labels import ClassicLabelsUpdater
        # Do NOT use ClassicLabelsUpdater.rollback: BasicUpdater always calls
        # CALENDAR.check_rollback_date(max_rollback_days=10), which fails for
        # historical start (e.g. 20050101). Call proceed_update with overwrite.
        wrap_update(
            ClassicLabelsUpdater.proceed_update ,
            'overwrite labels_ts (rtn+res columns)' ,
            start = start ,
            end = end ,
            overwrite = True ,
        )

    if do_factors:
        from src.res.factor.api import FactorUpdaterAPI
        wrap_update(
            FactorUpdaterAPI.Stock.fix ,
            f'fix stock factors {list(EXRET_FACTORS)}' ,
            factors = list(EXRET_FACTORS) ,
            start = start ,
            end = end ,
            timeout = -1 ,
        )
        wrap_update(FactorUpdaterAPI.export , 'export factor hierarchy table')

    if do_preprocess:
        from src.data.preprocess.task import PreProcessorTask
        for frame in ('fit' , 'predict'):
            wrap_update(
                PreProcessorTask.proceed_update ,
                f'reconstruct preprocess {list(PREPROCESS_KEYS)} frame={frame}' ,
                reconstruct = True ,
                frame = frame ,
                confirm = False ,
                data_types = list(PREPROCESS_KEYS) ,
                force_update = True ,
            )

    if do_cache:
        from src.data.util.classes.datacache import DataCache
        Logger.note(f'Purging ModuleData cache under {PATH.relative(PATH.datacache)}')
        DataCache.purge_all(confirm = True)
        Logger.success('DataCache purged')

    Logger.success('CNE5 downstream recalc finished')


if __name__ == '__main__':
    main()
