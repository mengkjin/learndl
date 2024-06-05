import numpy as np
import pandas as pd
from ashare_stock_id_sys.api import get_sys_id as get_stock_sys_id, get_all_sys_ids
from events_system.calendar_util import CALENDAR_UTIL
import os
import time
import datetime


DEFAULT_STORE_TYPE = 'ftr'

STORE_TYPE = dict()


def load_data_impl(src_path, scd=None, ecd=None, msg_tag='none', date_col="CalcDate", calendar_type='full', store_type='auto', required_dates=None):
    assert calendar_type in ['full', 'trade', 'required', 'unscheduled']
    data_files = os.listdir(src_path)
    if store_type == 'auto':
        exts = set([f.split('.')[1] for f in data_files])
        assert len(exts) == 1
        store_type = next(iter(exts))
    if scd is None:
        files = [f.split('.') for f in data_files]
        years = [f[0] for f in files if f[1] == store_type]
        scd = min(years) + '-01-01'
    if ecd is None:
        files = [f.split('.') for f in data_files]
        years = [f[0] for f in files if f[1] == store_type]
        ecd = max(years) + '-12-31'
    start_y, end_y = scd[:4], ecd[:4]
    df = list()
    for y in range(int(start_y), int(end_y) + 1):
        file = os.path.join(src_path, str(y) + '.' + store_type)
        if os.path.exists(file):
            if store_type == "ftr":
                y_data = pd.read_feather(file)
            elif store_type == "h5":
                y_data = pd.read_hdf(file, key='df')
            else:
                assert False
            if required_dates is not None:
                y_data = y_data[y_data[date_col].isin([d for d in required_dates if d[:4] == str(y)])]
            df.append(y_data)
    if df:
        df = pd.concat(df, axis=0)
        df = df[df[date_col].between(scd, ecd)].copy()
        if calendar_type == 'full':
            assert required_dates is None
            required_dates = CALENDAR_UTIL.get_ranged_dates(df[date_col].iloc[0], df[date_col].iloc[-1])
        elif calendar_type == 'trade':
            assert required_dates is None
            required_dates = CALENDAR_UTIL.get_ranged_trading_dates(df[date_col].iloc[0], df[date_col].iloc[-1])
        elif calendar_type == 'required':
            assert required_dates is not None
            pass
        elif calendar_type == 'unscheduled':
            required_dates = df[date_col].unique().tolist()
        else:
            assert False
        assert required_dates == df[date_col].unique().tolist()
        rtn = df
    else:
        assert False, "  error::{0}>>dv>>load>>can not find any data from {1} to {2} in path {3}".format(msg_tag,
                                                                                                         scd, ecd, src_path)
    return rtn


def load_data_in_one_year(path, year, msg_tag='none', calendar_type='full', store_type='auto'):
    if store_type == 'auto':
        exts = set([f.split('.')[1] for f in os.listdir(path)])
        assert len(exts) == 1
        store_type = next(iter(exts))
    file = os.path.join(path, year + '.' + store_type)
    assert os.path.exists(file), file
    if store_type == "ftr":
        rtn = pd.read_feather(file)
    elif store_type == "h5":
        rtn = pd.read_hdf(file, key="df")
    else:
        assert False, "  error::{0}>>dv>>load>>can not find any {1} data in year {2} in path {3}".format(
            msg_tag, store_type, year, path)
    assert rtn["CalcDate"].is_monotonic_increasing
    if calendar_type == 'full':
        required_dates = CALENDAR_UTIL.get_ranged_dates(rtn['CalcDate'].iloc[0], rtn['CalcDate'].iloc[-1])
    elif calendar_type == 'trade':
        required_dates = CALENDAR_UTIL.get_ranged_trading_dates(rtn['CalcDate'].iloc[0], rtn['CalcDate'].iloc[-1])
    elif calendar_type == 'unscheduled':
        required_dates = rtn['CalcDate'].unique().tolist()
    else:
        assert False
    assert required_dates == rtn['CalcDate'].unique().tolist()
    return rtn


def transfer_data(src_path, flds, des_path, store_type):
    src_df = load_data_impl(src_path)
    des_df = src_df[['CalcDate', 'Code'] + flds].copy()
    save(des_path, des_df, store_type)


def org_day_factor(x, all_sys_ids_, asset_col):
    cols = x.columns.drop(['CalcDate', asset_col, 'CodeID'])
    vals = pd.DataFrame(x[cols].to_numpy().astype(float), columns=cols)
    stk_sys_id = x.CodeID.values.astype(int)
    int_array = np.full_like(all_sys_ids_, -1)
    int_array[stk_sys_id] = np.arange(len(vals))
    return int_array, vals


class DataVendor:
    def __init__(self, src_path, msg_tag, asset_col='Code'):
        self.root_path = src_path
        self.msg_tag = msg_tag
        self.asset_col = asset_col
        self.orged_data = dict()  # key: calc_date
        self.raw_data = dict()  # key: year

    def _org_data(self, rslt_data):
        code_list = rslt_data[self.asset_col].unique()
        #
        stock_sys_id = get_stock_sys_id(code_list)
        assert np.all(np.array(stock_sys_id) >= 0)
        rslt_data = pd.merge(
            rslt_data,
            pd.DataFrame([code_list, stock_sys_id], index=[self.asset_col, 'CodeID']).T,
            how='left',
            on=[self.asset_col]
        )
        all_sys_ids = get_all_sys_ids()
        df = rslt_data.groupby(by=['CalcDate'], as_index=True).apply(
            func=org_day_factor, all_sys_ids_=all_sys_ids, asset_col=self.asset_col)
        self.orged_data.update(dict(zip(df.index, df.values.tolist())))

    def query_by_stk_sysid_on_calc_date(self, sys_id_list_, calc_date_):
        if calc_date_ not in self.orged_data:
            calc_y = calc_date_[:4]
            s = time.time()
            data_in_1year = load_data_impl(self.root_path, calc_y + '-01-01', calc_y + '-12-31', self.msg_tag)
            print("  status::{0}>>dv>>load year {1} daily data in secs {2}.".format(self.msg_tag, calc_y, str(time.time() - s)))
            self.raw_data[calc_y] = data_in_1year
            self._org_data(data_in_1year)
        data = self.orged_data.get(calc_date_)
        assert data is not None, "  error::{0}}>>dv>>has no data for {0} on date {1}.".format(self.msg_tag, calc_date_)
        int_array = data[0][sys_id_list_]
        return int_array, data[1].to_numpy().astype(float).T

    def load_data(self, scd, ecd, expected_cal_type='full', store_type="auto"):
        scy, ecy = int(scd[:4]), int(ecd[:4])
        rtn = list()
        for y in range(scy, ecy + 1):
            str_y = str(y)
            if str_y not in self.raw_data:
                s = time.time()
                self.raw_data[str_y] = load_data_in_one_year(self.root_path, str_y,
                                                             self.msg_tag, expected_cal_type, store_type)
                print("  status::{0}>>dv>>load year {1} daily data in secs {2}.".format(self.msg_tag,
                    str_y, str(time.time() - s)))
            rtn.append(self.raw_data[str_y])
        rtn = pd.concat(rtn, axis=0)
        rtn = rtn[rtn['CalcDate'].between(scd, ecd)].copy()
        if "op_date" in rtn.columns:
            rtn.drop(columns=["op_date"], inplace=True)
        if expected_cal_type == 'full':
            assert CALENDAR_UTIL.get_ranged_dates(scd, ecd) == rtn['CalcDate'].drop_duplicates().tolist(), \
                "  error::crosec_mem>>lackness of dates in {0}".format(self.root_path)
        elif expected_cal_type == 'trade':
            assert CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd) == rtn['CalcDate'].drop_duplicates().tolist(), \
                "  error::crosec_mem>>lackness of dates in {0}".format(self.root_path)
        elif expected_cal_type == 'unscheduled':
            pass
        else:
            assert False
        return rtn

    def load_data_by_datelist(self, date_list, expected_cal_type='full', store_type='auto'):
        year_list = sorted(list(set([d[:4] for d in date_list])))
        rtn = list()
        for y in year_list:
            if y not in self.raw_data:
                s = time.time()
                self.raw_data[y] = load_data_in_one_year(self.root_path, y, self.msg_tag, expected_cal_type, store_type)
                print("  status::{0}>>dv>>load year {1} daily data in secs {2}.".format(self.msg_tag,
                                                                                        y, str(time.time() - s)))
            y_date_list = [d for d in date_list if d[:4] == y]
            rtn.append(
                pd.merge(
                    pd.DataFrame(y_date_list, columns=['CalcDate']),
                    self.raw_data[y],
                    how='inner', on=['CalcDate']
                )
            )
        rtn = pd.concat(rtn, axis=0)
        assert rtn['CalcDate'].drop_duplicates().tolist() == date_list, self.msg_tag
        return rtn


def save(path, df, store_type=DEFAULT_STORE_TYPE, asset_col='Code', date_col="CalcDate"):
    df.sort_values(by=[date_col, asset_col], inplace=True)
    df = df.assign(op_date=datetime.datetime.now())
    if not os.path.exists(path):
        os.makedirs(path)
    years = df[date_col].str[:4]
    for y in years.unique():
        file = os.path.join(path, y + '.' + store_type)
        if os.path.exists(file):
            if store_type == "ftr":
                y_data = pd.read_feather(file)
            elif store_type == "h5":
                y_data = pd.read_hdf(file, key='df')
            else:
                assert False
            y_df = pd.concat((y_data, df[years == y]), axis=0)
            y_df.drop_duplicates(subset=[date_col, asset_col], keep='last', inplace=True)
        else:
            y_df = df[years == y].copy()
        y_df["op_date"].fillna(datetime.datetime.now(), inplace=True)
        y_df = y_df.sort_values(by=[date_col, asset_col]).reset_index(drop=True)
        if store_type == "ftr":
            y_df.to_feather(file)
        elif store_type == "h5":
            y_df.to_hdf(file, key='df', mode='w')
        else:
            assert False


def get_data_info(path):
    if not os.path.exists(path):
        os.makedirs(path)
    data_files = os.listdir(path)
    file_suffix_set = set([os.path.splitext(f)[1] for f in data_files])
    assert len(file_suffix_set) == 1, "  error::>>data_vendor>>get_data_info>>file suffix is not unique!"
    file_suffix = list(file_suffix_set)[0]
    full_year_files = [str(y) + file_suffix for y in range(2000, 2050)]
    data_files = sorted(list(set(data_files).intersection(full_year_files)))
    #
    rtn = dict()
    if data_files:
        file_years = [int(f[:4]) for f in data_files]
        rtn['start_year'] = file_years[0]
        rtn['end_year'] = file_years[-1]
        rtn['is_complete_in_range'] = file_years == list(range(file_years[0], file_years[-1] + 1))
        if file_suffix == '.h5':
            last_df = pd.read_hdf(os.path.join(path, data_files[-1]), key='df')
            frst_df = pd.read_hdf(os.path.join(path, data_files[0]), key='df')
        elif file_suffix == '.ftr':
            last_df = pd.read_feather(os.path.join(path, data_files[-1]))
            frst_df = pd.read_feather(os.path.join(path, data_files[0]))
        else:
            assert False, "  error::>>data_vendor>>get_data_info>>file suffix {0} is unknown!".format(file_suffix)
        last_date = last_df['CalcDate'].iloc[-1]
        frst_date = frst_df['CalcDate'].iloc[0]
        rtn['last_calc_date'] = last_date
        rtn['first_calc_date'] = frst_date
        # rtn['field'] = last_df.columns.drop(['CalcDate', 'Code'])
    else:
        rtn['start_year'] = None
        rtn['end_year'] = None
        rtn['is_complete_in_range'] = None
        rtn['first_calc_date'] = None
        rtn['last_calc_date'] = None
        # rtn['field'] = pd.Index([])
    return rtn


def transform_hdf_to_feather(src_path, des_path):
    assert os.path.exists(src_path)
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    data_files = os.listdir(src_path)
    file_suffix_set = list(set([os.path.splitext(f)[1] for f in data_files]))
    assert len(file_suffix_set) == 1 and file_suffix_set[0] == '.h5'
    for file_name in data_files:
        hdf_file_path = os.path.join(src_path, file_name)
        data = pd.read_hdf(hdf_file_path, key="df")
        ftr_file_path = os.path.join(des_path, os.path.splitext(file_name)[0] + '.ftr')
        data.to_feather(ftr_file_path)


def get_to_update_date_range(path, scd, ecd, calendar_type="full"):
    if os.path.exists(path) and os.listdir(path):
        data_files = os.listdir(path)
        file_suffix_set = set([os.path.splitext(f)[1] for f in data_files])
        assert len(file_suffix_set) == 1, "  error::>>data_vendor>>get_to_update_date_range>>file suffix is not unique!"
        file_suffix = list(file_suffix_set)[0]
        scy, ecy = int(scd[:4]), int(ecd[:4])
        saved_calc_dates = list()
        for y in range(scy, ecy + 1):
            file = os.path.join(path, str(y) + file_suffix)
            if os.path.exists(file):
                if file_suffix == ".h5":
                    date_list = pd.read_hdf(file, key="df")['CalcDate'].drop_duplicates().tolist()
                elif file_suffix == ".ftr":
                    date_list = pd.read_feather(file)['CalcDate'].drop_duplicates().tolist()
                else:
                    assert False, "  error::>>data_vendor>>get_to_update_date_range>>file suffix {0} is unknown!".format(file_suffix)
                saved_calc_dates.extend(date_list)
        if calendar_type == "full":
            to_calc_dates = set(CALENDAR_UTIL.get_ranged_dates(scd, ecd)).difference(set(saved_calc_dates))
        elif calendar_type == "trade":
            to_calc_dates = set(CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)).difference(set(saved_calc_dates))
        else:
            assert False
        if to_calc_dates:
            rtn = min(to_calc_dates), max(to_calc_dates)
        else:
            rtn = None
    else:
        rtn = scd, ecd
    return rtn