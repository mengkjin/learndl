from .configs import *
import pandas as pd
from .ashares.rules.trading import get_trading_data_filter
from .ashares.rules.barra import get_barra_filter
from .ashares.rules.st import get_st_filter
from .ashares.rules.bookvalue import get_bookvalue_filter
from .ashares.rules.index_comp import get_index_components_filter
from .ashares.rules.ipo_dates import get_n_listed_dates_filter
from .ashares.rules.to_patch_delist_earlier import get_early_delist
from events_system.calendar_util import CALENDAR_UTIL
from crosec_mem.data_vendor import save, get_to_update_date_range


def _calc_stkpool_data(root_path, scd, ecd):
    df_trading = get_trading_data_filter(root_path, scd, ecd)
    df_barra = get_barra_filter(root_path, scd, ecd)
    df_st = get_st_filter(scd, ecd)
    df_bv = get_bookvalue_filter(scd, ecd)
    df_index = get_index_components_filter(root_path, scd, ecd)
    df_n_ipo = get_n_listed_dates_filter(scd, ecd)
    df_early_kicked = get_early_delist(scd, ecd)
    df = pd.merge(df_barra, df_trading, how='left', on=['CalcDate', 'Code'])
    print("  warning::stock pool>>generator>>must include index member in barra")
    df = pd.merge(df, df_bv, how="left", on=['CalcDate', 'Code'])
    df = pd.merge(df, df_st, how="left", on=['CalcDate', 'Code'])
    df['is_st_within_90d'] = df['is_st_within_90d'].fillna(0.0)
    df = pd.merge(df, df_early_kicked, how='left', on=['CalcDate', 'Code'])
    df['is_earlier_kicked'] = df['is_earlier_kicked'].fillna(0.0)
    df = pd.merge(df, df_index, how="left", on=['CalcDate', 'Code'])
    df[['000300.SH', '000905.SH', '000852.SH', '932000.CSI', '000906.SH', '931865.CSI']] = \
        df[['000300.SH', '000905.SH', '000852.SH', '932000.CSI', '000906.SH', '931865.CSI']] > 1.0e-5
    rtn = pd.merge(df, df_n_ipo, how='left', on=['CalcDate', 'Code'])
    rtn = rtn[rtn['Code'].str[-2:] != 'BJ'].copy()
    assert rtn['CalcDate'].drop_duplicates().tolist() == CALENDAR_UTIL.get_ranged_dates(scd, ecd)
    return rtn


def gen_stkpool_data(root_path, scd, ecd):
    assert scd >= '2003-01-01'
    path = get_data_path(root_path, 'ashares')
    to_update_date_range = get_to_update_date_range(path, scd, ecd)
    if to_update_date_range is not None:
        to_update_scd, to_update_ecd = to_update_date_range
        rtn = _calc_stkpool_data(root_path, to_update_scd, to_update_ecd)
        #
        save(path, rtn, store_type='ftr')
        print("  status::ashare_stkpool>>generator>>generate stock pool data from '{0}' to '{1}'.".format(to_update_scd, to_update_ecd))
    else:
        print("  status::ashare_stkpool>>generator>>stock pool data already exists from {0} to {1}.".format(scd, ecd))