"""
Build ``factors.csv``: one row per implemented min_chars column.

Not a ``BasicCustomUpdater`` (``_`` prefix).  Run:

    uv run python -c "from src.data.update.custom.min_chars._catalog import write_csv; write_csv()"
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.data.update.custom.min_chars._common import DB_SRC
from src.data.update.custom.min_chars.daily import (
    APPROX_FEATURES ,
    DAILY_FLOW ,
    DAILY_HF_5MIN ,
    DAILY_HF_CORR ,
    DAILY_HF_LIQ ,
    DAILY_HF_MOM ,
    DAILY_HF_VOL ,
    DAILY_PX ,
    DAILY_RV ,
    OUTPUT_COLUMNS ,
    SESSION_STEMS ,
    _session_cols ,
)
from src.data.update.custom.min_chars.rolling import (
    POOL_COLUMNS ,
    ROLL_COLUMNS ,
    TRAIL_SPECS ,
)
from src.data.update.custom.min_chars.tagged import TAG_COLUMNS , TAGS , TAG_METRICS

CSV_PATH = Path(__file__).with_name('factors.csv')
CSV_FIELDS = (
    'name' , 'stage' , 'db_src' , 'db_key' , 'family' , 'window' , 'agg' ,
    'daily_src' , 'hf_factor' , 'formula' ,
)

_HF_DAILY : dict[str , str] = {
    'mkt_beta' : 'inday_mkt_beta;inday_mkt_beta_std' ,
    'mkt_corr' : 'inday_mkt_corr;inday_mkt_corr_std' ,
    'ret_autocorr' : 'inday_ret_autocorr;inday_ret_autocorr_std' ,
    'vol_autocorr' : 'inday_vol_autocorr;inday_vol_autocorr_std' ,
    'vol_retlag_corr' : 'inday_vol_ret1_corr;inday_vol_ret1_corr_std' ,
    'vol_vwap_corr' : 'inday_vol_vwap_corr;inday_vol_vwap_corr_std' ,
    'ret_std' : 'inday_std_1min;mom_high_pstd;vol_high_std' ,
    'ret_std5' : 'inday_std_5min' ,
    'ret_skew' : 'inday_skewness_1min' ,
    'ret_skew5' : 'inday_skewness_5min' ,
    'ret_kurt' : 'inday_kurt_1min' ,
    'ret_kurt5' : 'inday_kurt_5min' ,
    'ret_vardown' : 'inday_vardown_1min' ,
    'ret_vardown5' : 'vardown_intra5min' ,
    'vol_cv' : 'inday_vol_std_1min;inday_vol_coefvar;mom_high_volcv' ,
    'vol_cv5' : 'inday_vol_std_5min' ,
    'ret_topk_mean' : 'inday_err_ret' ,
    'ret_maxdd' : 'inday_maxdd' ,
    'vol_std' : 'inday_vol_utd' ,
    'smart_money' : 'inday_smart_money' ,
    'stupid_money' : 'inday_stupid_money' ,
    'vol_end15_share' : 'inday_vol_end15min' ,
    'vol_open5_share' : 'inday_vol_st5min' ,
    'vol_highrank_share' : 'inday_volpct_phigh' ,
    'vol_lowrank_share' : 'inday_volpct_plow' ,
    'vol_highdev_share' : 'inday_volpct_devhigh' ,
    'ret_am' : 'inday_amap_orig' ,
    'ret_pm' : 'inday_amap_orig' ,
    'conf_persist' : 'inday_conf_persist;inday_regain_conf_persist' ,
    'high_time' : 'inday_high_time' ,
    'incvol_ret' : 'inday_incvol_mom' ,
    'vwap_trend' : 'inday_trend_avg;inday_trend_std' ,
    'vwap_hlvol' : 'inday_vwap_diff_hlvol' ,
}

_HF_TRAIL : dict[str , str] = {
    'mkt_beta_ma20' : 'inday_mkt_beta' ,
    'mkt_corr_ma20' : 'inday_mkt_corr' ,
    'ret_autocorr_ma20' : 'inday_ret_autocorr' ,
    'vol_autocorr_ma20' : 'inday_vol_autocorr' ,
    'vol_retlag_corr_ma20' : 'inday_vol_ret1_corr' ,
    'vol_vwap_corr_ma20' : 'inday_vol_vwap_corr' ,
    'mkt_beta_std20' : 'inday_mkt_beta_std' ,
    'mkt_corr_std20' : 'inday_mkt_corr_std' ,
    'ret_autocorr_std20' : 'inday_ret_autocorr_std' ,
    'vol_autocorr_std20' : 'inday_vol_autocorr_std' ,
    'vol_retlag_corr_std20' : 'inday_vol_ret1_corr_std' ,
    'vol_vwap_corr_std20' : 'inday_vol_vwap_corr_std' ,
    'ret_topk_mean_ma20' : 'inday_err_ret' ,
    'ret_std_ma20' : 'inday_std_1min' ,
    'ret_std5_ma20' : 'inday_std_5min' ,
    'ret_skew_ma20' : 'inday_skewness_1min' ,
    'ret_skew5_ma20' : 'inday_skewness_5min' ,
    'ret_kurt_ma20' : 'inday_kurt_1min' ,
    'ret_kurt5_ma20' : 'inday_kurt_5min' ,
    'ret_vardown_ma20' : 'inday_vardown_1min' ,
    'ret_vardown5_ma20' : 'vardown_intra5min' ,
    'vol_cv_ma20' : 'inday_vol_std_1min;inday_vol_coefvar' ,
    'vol_cv5_ma20' : 'inday_vol_std_5min' ,
    'ret_maxdd_max5' : 'inday_maxdd' ,
    'smart_money_ma20' : 'inday_smart_money' ,
    'stupid_money_ma20' : 'inday_stupid_money' ,
    'vol_std_cv20' : 'inday_vol_utd' ,
    'vol_end15_share_ma20' : 'inday_vol_end15min' ,
    'vol_open5_share_ma20' : 'inday_vol_st5min' ,
    'vol_highrank_share_ma20' : 'inday_volpct_phigh' ,
    'vol_lowrank_share_ma20' : 'inday_volpct_plow' ,
    'vol_highdev_share_ma20' : 'inday_volpct_devhigh' ,
    'high_time_ma20' : 'inday_high_time' ,
    'incvol_ret_sum20' : 'inday_incvol_mom' ,
    'vwap_trend_ma20' : 'inday_trend_avg' ,
    'vwap_trend_std20' : 'inday_trend_std' ,
    'vwap_hlvol_ma20' : 'inday_vwap_diff_hlvol' ,
    'conf_persist_ma20' : 'inday_conf_persist' ,
    'conf_persist_std20' : 'inday_conf_persist' ,
}

_DAILY_FORMULA : dict[str , str] = {
    'amt' : 'sum(amount)' ,
    'twap' : 'mean(close)' ,
    'vwap' : 'sum(amount)/sum(volume); fill last close' ,
    'bwap' : 'sum(buy_amt*px)/sum(buy_amt)' ,
    'swap' : 'sum(sell_amt*px)/sum(sell_amt)' ,
    'bamt' : 'sum(amount) where ret>0 (flat split 50/50)' ,
    'samt' : 'sum(amount) where ret<0 (flat split 50/50)' ,
    'ret_path' : '(prod(1+ret)-1)*100' ,
    'bopct' : 'bamt/(bamt+samt)*100' ,
    'amt_ca' : 'sum(amount) for minute>=237' ,
    'ret_std' : 'std(ret)' ,
    'ret_skew' : 'skew(ret), n>=3' ,
    'ret_kurt' : 'kurtosis(ret, fisher=False), n>=3' ,
    'vol_std' : 'std(volume)' ,
    'vol_hhi' : 'n*sum(volume^2)/sum(volume)^2' ,
    'ret_jump' : 'ret*100 of max |ret| bar' ,
    'bopct_h1' : 'bopct of sess=0 (09:30-10:00)' ,
    'ret_topk_mean' : 'mean(ret) above top_k(5) threshold' ,
    'ret_maxdd' : 'max(1-low/cum_high)' ,
    'ret_vardown' : 'std(ret|ret<0)/std(ret)' ,
    'vol_cv' : 'std(volume)/mean(volume)' ,
    'ret_std5' : 'std(ret) on minute//5 bars' ,
    'ret_skew5' : 'skew(ret) on 5-min bars' ,
    'ret_kurt5' : 'Pearson kurtosis on 5-min bars' ,
    'ret_vardown5' : 'downside std ratio on 5-min bars' ,
    'vol_cv5' : 'volume CV on 5-min bars' ,
    'mkt_beta' : 'corr(ret,mkt)*std(ret)/std(mkt)' ,
    'mkt_corr' : 'corr(ret,mkt)' ,
    'ret_autocorr' : 'corr(ret, ret.shift1)' ,
    'vol_autocorr' : 'corr(volume, volume.shift1)' ,
    'vol_retlag_corr' : 'corr(volume, ret.shift1)' ,
    'vol_vwap_corr' : 'corr(volume, px)' ,
    'smart_money' : 'volume share of next-ret rank top 10%' ,
    'stupid_money' : 'volume share of next-ret rank bottom 10%' ,
    'vol_end15_share' : 'volume share minute>=225' ,
    'vol_open5_share' : 'volume share minute<5' ,
    'vol_highrank_share' : 'volume share of volume-rank>=90%' ,
    'vol_lowrank_share' : 'volume share of volume-rank<=10%' ,
    'vol_highdev_share' : 'volume share of |dclose|/max>=90%' ,
    'ret_am' : 'prod(1+ret)-1 for minute<120' ,
    'ret_pm' : 'prod(1+ret)-1 for minute>=120' ,
    'conf_persist' : 'median(minute|down)-median(minute|up)' ,
    'high_time' : 'median(minute) of top-5 high' ,
    'incvol_ret' : 'sum(ret) where volume>shift1' ,
    'vwap_trend' : 'corr(px,minute)*std(px)/std(minute)' ,
    'vwap_hlvol' : '(vwap_highvol-vwap_lowvol)/avg' ,
}


def _family(name : str) -> str:
    if name in ('date' , 'secid' , 'n'):
        return 'key'
    if name in DAILY_PX or name.startswith('amt') or name in ('twap' , 'vwap'):
        return 'px'
    if name in DAILY_FLOW or name.startswith('bamt') or name.startswith('samt') or name.startswith('bopct'):
        return 'flow'
    if name in DAILY_RV or name in DAILY_HF_VOL or name in DAILY_HF_5MIN:
        return 'rv'
    if name in DAILY_HF_CORR or 'corr' in name or name.startswith('mkt_'):
        return 'corr'
    if name in DAILY_HF_LIQ or 'share' in name or 'money' in name:
        return 'liq'
    if name in DAILY_HF_MOM or name.startswith('conf_') or name.startswith('vwap_') or name.startswith('incvol') or name.startswith('high_time') or name.startswith('ret_am') or name.startswith('ret_pm'):
        return 'mom'
    if name.endswith('h') and name[-2].isdigit():
        return 'session'
    if '_p0' in name or '_p5' in name or '_p9' in name or name.endswith('_p50') or '_pool_' in name:
        return 'pool'
    if name.endswith(('_ma20' , '_std20' , '_cv20' , '_sum20' , '_max5')):
        return 'trail'
    if any(name.endswith(f'_{tag[0]}') or f'_{tag[0]}' in name for tag in TAGS):
        return 'tag'
    return 'other'


def _session_formula(name : str) -> str:
    for stem in SESSION_STEMS:
        if name.startswith(stem) and name[len(stem):].endswith('h'):
            k = name[len(stem):-1]
            if k.isdigit():
                base = _DAILY_FORMULA.get(stem , stem)
                return f'{base} within sess={int(k)-1} ({name[-2:]})'
    return name


def catalog_rows() -> list[dict[str , Any]]:
    """All implemented output columns as catalog rows."""
    rows : list[dict[str , Any]] = []

    for name in OUTPUT_COLUMNS:
        if name in ('date' , 'secid'):
            continue
        formula = _DAILY_FORMULA.get(name , _session_formula(name))
        family = 'session' if name in _session_cols(SESSION_STEMS) else _family(name)
        if name in APPROX_FEATURES:
            family = 'approx'
        rows.append({
            'name' : name ,
            'stage' : 'daily' ,
            'db_src' : DB_SRC ,
            'db_key' : 'min_chars' ,
            'family' : family ,
            'window' : '1' ,
            'agg' : 'same_day' ,
            'daily_src' : '' ,
            'hf_factor' : _HF_DAILY.get(name , '') ,
            'formula' : formula ,
        })

    for name in ROLL_COLUMNS:
        if name in ('date' , 'secid'):
            continue
        if name in POOL_COLUMNS:
            if name == 'n':
                formula = 'count of pooled 1-min bars in window'
            elif '_pool_' in name:
                stem , _ , stat = name.partition('_pool_')
                formula = f'{stat} of pooled 1-min {stem} over {20} min dates'
            else:
                stem , _ , q = name.partition('_p')
                formula = f'quantile 0.{q} of pooled 1-min {stem}'
            rows.append({
                'name' : name ,
                'stage' : 'roll_pool' ,
                'db_src' : DB_SRC ,
                'db_key' : 'min_chars_roll' ,
                'family' : 'pool' ,
                'window' : '20' ,
                'agg' : 'pool_minutes' ,
                'daily_src' : '' ,
                'hf_factor' : '' ,
                'formula' : formula ,
            })
            continue
        spec = next((s for s in TRAIL_SPECS if s[3] == name) , None)
        if spec is None:
            continue
        src , how , win , out = spec
        rows.append({
            'name' : out ,
            'stage' : 'roll_trail' ,
            'db_src' : DB_SRC ,
            'db_key' : 'min_chars_roll' ,
            'family' : 'trail' ,
            'window' : str(win) ,
            'agg' : how ,
            'daily_src' : src ,
            'hf_factor' : _HF_TRAIL.get(out , '') ,
            'formula' : f'{how} of daily {src} over {win} days' ,
        })

    tag_formula = {
        'ret_path' : '(prod(1+ret)-1)*100 on tagged minutes; null if none' ,
        'ret_mean' : 'mean(ret) on tagged minutes; null if none' ,
        'amt_share' : 'sum(amount_tagged)/sum(amount); 0 if none' ,
    }
    tag_cond = {t[0] : f'{t[1]} {t[2]} {t[3]}' for t in TAGS}
    for name in TAG_COLUMNS:
        if name in ('date' , 'secid'):
            continue
        metric , _ , tag = name.partition('_')
        # names are ret_path_rethi99 / ret_mean_rethi99 / amt_share_rethi99
        for m in TAG_METRICS:
            prefix = f'{m}_'
            if name.startswith(prefix):
                tag = name[len(prefix):]
                rows.append({
                    'name' : name ,
                    'stage' : 'tag' ,
                    'db_src' : DB_SRC ,
                    'db_key' : 'min_chars_tag' ,
                    'family' : 'tag' ,
                    'window' : '20+1' ,
                    'agg' : 'tag_today' ,
                    'daily_src' : tag_cond.get(tag , '') ,
                    'hf_factor' : '' ,
                    'formula' : f'{tag_formula[m]} | {tag_cond.get(tag , tag)}' ,
                })
                break

    return rows


def write_csv(path : Path = CSV_PATH) -> Path:
    """Write ``factors.csv`` next to this module."""
    rows = catalog_rows()
    with path.open('w' , newline = '' , encoding = 'utf-8') as f:
        w = csv.writer(f)
        w.writerow(list(CSV_FIELDS))
        for row in rows:
            w.writerow([row[k] for k in CSV_FIELDS])
    return path
