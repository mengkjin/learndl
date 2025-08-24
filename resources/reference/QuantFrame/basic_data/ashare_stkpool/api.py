from .configs import *
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from crosec_mem.data_vendor import DataVendor


DATA_VENDOR = dict()

default_pool_configs = {
    'universe':
        {
            'ma_clsprc': 2.0,
            'book_value': 1000000000.0,
            'ma_amt_pctile': 0.3,
            'ma_ttlcap_pctile': 0.3,
            'n_listed': 365,
            'with_st': 0.5,
            'has_barra': 0.5,
            'is_earlier_kicked': 0.5,
            'index_mem': ['000300.SH']
        },
    'universe_new':
        {
            'ma_clsprc': 2.0,
            'book_value': 10000000.0,
            'ma_amt_pctile': 0.1,
            'ma_ttlcap_pctile': 0.1,
            'n_listed': 365,
            'with_st': 0.5,
            'has_barra': 0.5,
            'is_earlier_kicked': 0.5,
        },
    'smpl_univ':
        {
            'n_listed': 365,
            'with_st': 0.5,
            'has_barra': 0.5,
            'is_earlier_kicked': 0.5,
        },
    'nonstonly':
        {
            'with_st': 0.5,
            'has_barra': 0.5,
            'is_earlier_kicked': 0.5,
        },
    '000300.SH':
        {
            'index_mem': ['000300.SH'],
            'ma_clsprc': 10000000.0  # trick
        },
    '000905.SH':
        {
            'index_mem': ['000905.SH'],
            'ma_clsprc': 10000000.0  # trick
        },
    '000852.SH':
        {
            'index_mem': ['000852.SH'],
            'ma_clsprc': 10000000.0  # trick
        },
    '932000.CSI':
        {
            'index_mem': ['932000.CSI'],
            'ma_clsprc': 10000000.0  # trick
        },
    '000906.SH':
        {
            'index_mem': ['000906.SH'],
            'ma_clsprc': 10000000.0  # trick
        },    
    '931865.CSI':
        {
            'index_mem': ['931865.CSI'],
            'ma_clsprc': 10000000.0  # trick
        }
}


def filter_stock_pool(df,
                      ma_amt=None, ma_clsprc=None, ma_ttlcap=None, book_value=None, ma_amt_pctile=None,
                      ma_ttlcap_pctile=None, n_listed=None, with_st=None, has_barra=None, is_earlier_kicked=None,
                      index_mem=None):
    flg = pd.Series([True] * df.shape[0], index=df.index)
    if ma_amt is not None:
        flg = flg & (df['ma_amt'] >= ma_amt)
    if ma_ttlcap is not None:
        flg = flg & (df['ma_ttlcap'] >= ma_ttlcap)
    if book_value is not None:
        flg = flg & (df['book_value'] >= book_value)
    if ma_clsprc is not None:
        flg = flg & (df['ma_clsprc'] >= ma_clsprc)
    if ma_amt_pctile is not None:
        flg = flg & (df['ma_amt_pctile'] >= ma_amt_pctile)
    if ma_ttlcap_pctile is not None:
        flg = flg & (df['ma_ttlcap_pctile'] >= ma_ttlcap_pctile)
    if n_listed is not None:
        flg = flg & (df['n_listed_date'] >= n_listed)
    if with_st is not None:
        flg = flg & (df['is_st_within_90d'] < with_st)
    if has_barra is not None:
        flg = flg & (df['has_barra'] > has_barra)
    if is_earlier_kicked is not None:
        flg = flg & (df['is_earlier_kicked'] < is_earlier_kicked)
    if index_mem is not None:  # condition is at end
        assert len(index_mem) >= 1
        flg = flg | (df[index_mem].sum(axis=1) > 1.0e-5)
    rtn = df.loc[flg, ['CalcDate', 'Code']].copy()
    return rtn


def get_path(root_path, pool_type):
    if pool_type in list(default_pool_configs.keys()) + ['ashares']:
        path = get_data_path(root_path, 'ashares')
    else:
        path = os.path.join(root_path, pool_type)
    return path


def load_stkpool_data_prd(root_path, scd, ecd, pool_type='ashares', with_ext_info=False, **kwargs):
    assert pool_type != 'ashare', "  TODO:for debuging."
    key = root_path + '@' + pool_type
    path = get_path(root_path, pool_type)
    if key not in DATA_VENDOR:
        DATA_VENDOR[key] = DataVendor(path, "stockpool")
    rtn = DATA_VENDOR[key].load_data(scd, ecd, expected_cal_type='full')
    if pool_type in list(default_pool_configs.keys()):
        cfg = default_pool_configs[pool_type]
        rtn = filter_stock_pool(rtn, **cfg)
    elif pool_type == 'ashares':
        rtn = filter_stock_pool(rtn, **kwargs)
    #
    loaded_dates = rtn['CalcDate'].unique().tolist()
    target_dates = CALENDAR_UTIL.get_ranged_dates(scd, ecd)
    assert loaded_dates == target_dates, \
        "  status::ashare_stkpool>>api>>load pool dates for pool type: " \
        "{0} from {1} to {2} are not enough.".format(pool_type, scd, ecd)
    return rtn


# def load_stkpool_data_by_dates1(root_path, dates, pool_type='universe', ext_info_list=None, **kwargs):
#     assert ext_info_list is None or isinstance(ext_info_list, list)
#     src_path = get_path(root_path, pool_type)
#     pool_df = load_data_by_dates(src_path, dates, store_type='auto', date_col='CalcDate')
#     if ext_info_list is None:
#         rtn = pd.merge(
#             pd.DataFrame(dates, columns=['CalcDate']),
#             pool_df[['CalcDate', 'Code']], how='left', on=['CalcDate']
#         )
#     else:
#         rtn = pd.merge(
#             pd.DataFrame(dates, columns=['CalcDate']),
#             pool_df[['CalcDate', 'Code'] + ext_info_list], how='left', on=['CalcDate']
#         )
#     return rtn


def load_stkpool_data_by_dates(root_path, dates, pool_type='universe', ext_info_list=None, **kwargs):
    assert ext_info_list is None or isinstance(ext_info_list, list)
    scd, ecd = dates[0], dates[-1]
    pool_df = load_stkpool_data_prd(root_path, scd, ecd, pool_type, **kwargs)
    if ext_info_list is None:
        rtn = pd.merge(
            pd.DataFrame(dates, columns=['CalcDate']),
            pool_df[['CalcDate', 'Code']], how='left', on=['CalcDate']
        )
    else:
        rtn = pd.merge(
            pd.DataFrame(dates, columns=['CalcDate']),
            pool_df[['CalcDate', 'Code'] + ext_info_list], how='left', on=['CalcDate']
        )
    return rtn


def remove_outpool_data(root_path, df, pool_type, remove_type='inner', **kwargs):
    assert 'CalcDate' in df.columns and 'Code' in df.columns
    query_list = df['CalcDate'].drop_duplicates().tolist()
    assert sorted(query_list) == query_list
    univ_pool = load_stkpool_data_by_dates(root_path, query_list, pool_type, **kwargs)
    if remove_type == 'inner':
        rtn = pd.merge(df, univ_pool, how='inner', on=['CalcDate', 'Code'], sort=True)
    elif remove_type == 'left':
        assert 'in_pool' not in df.columns
        rtn = pd.merge(df, univ_pool.assign(in_pool=1), how='left', on=['CalcDate', 'Code'])
    else:
        assert False
    assert df['CalcDate'].drop_duplicates().tolist() == rtn['CalcDate'].drop_duplicates().tolist(), \
        '  error::ashare_stkpool>>dates not match. try remove_type = "left".'
    return rtn


def merge_to_stock_pool(root_path, df, pool_type, ext_info_list=None, date_code_cols=None, **kwargs):
    if date_code_cols is None:
        assert isinstance(df.index, pd.MultiIndex) and len(df.index.names) == 2
        date_col, code_col = df.index.names
        assert df.index.get_level_values(date_col).is_monotonic_increasing
        query_list = df.index.get_level_values(date_col).drop_duplicates().tolist()
    else:
        date_col, code_col = date_code_cols
        assert df[date_col].is_monotonic_increasing
        query_list = df[date_col].drop_duplicates().tolist()
    univ_pool = load_stkpool_data_by_dates(root_path, query_list, pool_type, ext_info_list=ext_info_list, **kwargs)
    rtn = pd.merge(univ_pool.rename(columns={'CalcDate': date_col, 'Code': code_col}, errors='raise'), df, how='left', on=[date_col, code_col], sort=True)
    if date_code_cols is None:
        rtn.set_index([date_col, code_col], inplace=True)
    return rtn


def get_data_info(root_path, pool_type):
    data_path = get_data_path(root_path, pool_type)
    files = os.listdir(data_path)
    files = [f for f in files if os.path.splitext(f)[1] == '.h5']
    files.sort()
    first_file, last_file = files[0], files[-1]
    first_df = pd.read_hdf(os.path.join(data_path, first_file), key='df')
    first_calc_date = first_df['CalcDate'].iloc[0]
    last_df = pd.read_hdf(os.path.join(data_path, last_file), key='df')
    last_calc_date = last_df['CalcDate'].iloc[-1]
    rtn = {
        'first_calc_date': first_calc_date,
        'last_calc_date': last_calc_date
    }
    return rtn