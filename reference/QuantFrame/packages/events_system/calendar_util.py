import numpy as np
import pandas as pd
import copy
# from common_tools.database_util import DataBaseConnector
from collections import OrderedDict
import os
from datetime import datetime
import calendar


TLDB = "basicinfo_ro"


class CalendarUtil:

    INFINITY = 99999

    def __init__(self):
        self.calendar = None
        self.t_calendar = None
        self.index_dict = None
        self.calendar_date_index_dict = None
        self.ymd_dict = None

        holidays = pd.read_csv(os.path.join(os.path.dirname(__file__), 'holiday.csv'))
        #
        now_date = datetime.now().strftime("%Y-%m-%d")
        if now_date[5:] >= '12-15' and holidays['Date'].iloc[-1][:4] == now_date[:4]:
            print(" warning::>>calendar_util>>file holiday.csv need update!")
        #
        start_date, end_date = holidays['Date'].iloc[0][:4] + '-01-01', holidays['Date'].iloc[-1][:4] + '-12-31'
        all_dates = pd.date_range(start_date, end_date).tolist()
        weekdays = [d.weekday() + 1 for d in all_dates]
        eomdays = [datetime(d.year, d.month, calendar.monthrange(d.year, d.month)[1]).strftime('%Y-%m-%d') for d in all_dates]
        iso_week_nums = [(d.isocalendar()[0], d.isocalendar()[1]) for d in all_dates]
        all_dates = [d.strftime('%Y-%m-%d') for d in all_dates]

        dates_info = pd.DataFrame([all_dates, weekdays, eomdays, iso_week_nums], index=['date', 'weekday', 'eom', 'iso_week_num']).T
        dates_info['holiday'] = dates_info['date'].isin(holidays['Date'])
        dates_info['isTrading'] = ((dates_info['weekday'] < 6) & (~dates_info['holiday'])) * 1
        dates_info['year'] = dates_info['date'].str[:4]
        t_eow_dates = dates_info[dates_info['isTrading'] == 1].groupby(['iso_week_num'])['date'].last()
        dates_info['is_t_eow'] = dates_info['date'].isin(t_eow_dates) * 1
        dates_info['ym'] = dates_info['date'].str[:7]
        t_eom_dates = dates_info[dates_info['isTrading'] == 1].groupby(['ym'])['date'].last()
        dates_info['is_t_eom'] = dates_info['date'].isin(t_eom_dates) * 1
        self._init_index_dict(list(dates_info["date"]), list(dates_info["isTrading"]))
        self.t_eom = dates_info[dates_info['is_t_eom'] == 1]["date"].tolist()
        self.t_eow = dates_info[dates_info['is_t_eow'] == 1]["date"].tolist()
        self.t_eoh = [d for d in self.t_eom if d[5:7] == '06' or d[5:7] == '12']

    def _init_index_dict(self, dates_, is_tradings_):
        assert set(is_tradings_) == {0, 1}
        ixs = np.cumsum(is_tradings_)
        ixs -= 1
        flag = np.logical_not(ixs < 0)
        dates = np.array(dates_)[flag]
        is_tradings = np.array(is_tradings_)[flag]
        ixs = ixs[flag]
        self.index_dict = dict(zip(dates, ixs))
        self.calendar_date_index_dict = dict(zip(dates, list(range(len(dates)))))
        # self.calendar = list(dates)  # warning: fuck np.str_ and str
        # self.t_calendar = list(dates[is_tradings == 1])
        self.calendar = [str(d) for d in dates]
        self.t_calendar = [str(d) for d in dates[is_tradings == 1]]
        self.ymd_dict = OrderedDict()
        for date in self.calendar:
            y = date[:4]
            m = date[5:7]
            d = date[8:]
            if y not in self.ymd_dict:
                self.ymd_dict[y] = OrderedDict()
            if m not in self.ymd_dict[y]:
                self.ymd_dict[y][m] = list()
            self.ymd_dict[y][m].append(d)

    def get_latest_n_trading_dates(self, date_: str, n_: int):
        if n_ > 0:
            ix = self.index_dict.get(date_, None)
            assert ix is not None, date_
            assert ix >= n_ - 1, date_ + ":" + str(n_)
            dates = self.t_calendar[ix - n_ + 1: ix + 1]
        elif n_ < 0:
            abs_n = -n_
            ix = self.index_dict.get(date_, None)
            assert ix is not None
            if self.t_calendar[ix] < date_:
                assert len(self.t_calendar[ix + 1:]) >= abs_n
                dates = self.t_calendar[ix + 1: ix + abs_n + 1]
            elif self.t_calendar[ix] == date_:
                assert len(self.t_calendar[ix:]) >= abs_n
                dates = self.t_calendar[ix: ix + abs_n]
            else:
                assert False
        else:
            assert False, "n_ can't be zero"
        return dates

    def get_latest_n_dates(self, date_: str, n: int):
        assert n > 0
        ix = self.calendar_date_index_dict.get(date_, None)
        assert ix is not None
        assert ix >= n - 1
        dates = self.calendar[ix - n + 1: ix + 1]
        return dates

    def get_fwd_n_dates(self, date_: str, n_: int, is_include_itself=False):
        assert n_ > 0
        ix = self.calendar_date_index_dict.get(date_, None)
        assert ix is not None
        if is_include_itself:
            dates = self.calendar[ix: ix + n_]
        else:
            dates = self.calendar[ix + 1: ix + n_ + 1]
        return dates

    def get_fwd_n_trading_dates(self, date_: str, n: int, self_inclusive=False):
        assert n > 0
        ix = self.index_dict.get(date_, None)
        assert ix is not None
        if self_inclusive and self.t_calendar[ix] == date_:
            dates = self.t_calendar[ix: ix + n]
        else:
            dates = self.t_calendar[ix + 1: ix + n + 1]
        return dates

    def get_ranged_trading_dates(self, start_date_: str, end_date_: str, including_latest=False):
        start_ix = self.index_dict.get(start_date_, None)
        assert start_ix is not None
        end_ix = self.index_dict.get(end_date_, None)
        assert end_ix is not None
        assert self.t_calendar[start_ix] <= start_date_
        if including_latest:
            dates = self.t_calendar[start_ix: end_ix + 1]
        else:
            if self.t_calendar[start_ix] < start_date_:
                dates = self.t_calendar[start_ix + 1: end_ix + 1]
            else:
                dates = self.t_calendar[start_ix: end_ix + 1]
        return dates

    def get_ranged_dates(self, start_date_: str, end_date_: str):
        start_ix = self.calendar_date_index_dict.get(start_date_, None)
        assert start_ix is not None
        end_ix = self.calendar_date_index_dict.get(end_date_, None)
        assert end_ix is not None
        assert end_ix + 1 <= len(self.calendar)
        dates = self.calendar[start_ix: (end_ix + 1)]
        return dates

    def get_next_trading_dates(self, date_list_: list, inc_self_if_is_trdday, n=1):
        assert n > 0
        rtn = list()
        for date in date_list_:
            ix = self.index_dict.get(date, None)
            assert ix is not None
            assert ix < len(self.t_calendar) - 1
            if inc_self_if_is_trdday:
                if date == self.t_calendar[ix]:
                    rtn.append(self.t_calendar[ix + n - 1])
                else:
                    rtn.append(self.t_calendar[ix + n])
            else:
                rtn.append(self.t_calendar[ix + n])
        return rtn

    def get_next_dates(self, date_list_: list, n=1):
        assert n > 0
        rtn = list()
        for date in date_list_:
            ix = self.calendar_date_index_dict.get(date, None)
            assert ix is not None
            assert ix < len(self.calendar) - n
            rtn.append(self.calendar[ix + n])
        return rtn

    def get_last_trading_dates(self, date_list_: list, inc_self_if_is_trdday, raise_error_if_nan=True, n=1):
        assert n > 0
        rtn = list()
        for date in date_list_:
            if date == 'nan' or date is None:
                if raise_error_if_nan:
                    assert False, "  error::calendar_util>>nan or null date found."
                else:
                    rtn.append(date)
            else:
                to_shift = n - 1
                ix = self.index_dict.get(date, None)
                assert ix is not None
                if inc_self_if_is_trdday:
                    rtn.append(self.t_calendar[ix - to_shift])
                else:
                    if date == self.t_calendar[ix]:
                        assert ix >= 1 + to_shift
                        rtn.append(self.t_calendar[ix - 1 - to_shift])
                    else:
                        rtn.append(self.t_calendar[ix - to_shift])
        return rtn

    def get_last_dates(self, date_list_: list, raise_error_if_nan=True, n=1):
        assert n >= 1 and isinstance(date_list_, (list, tuple))
        rtn = list()
        for date in date_list_:
            if date is None or date == 'nan':
                if raise_error_if_nan:
                    assert False, "  error::calendar_util>>nan or null date found."
                else:
                    rtn.append(date)
            else:
                ix = self.calendar_date_index_dict.get(date, None)
                assert ix is not None and ix >= n
                rtn.append(self.calendar[ix - n])
        return rtn

    def get_last_trading_eoms(self, date_list_: list, inc_self_if_is_trdday):
        assert inc_self_if_is_trdday
        all_dates = set(date_list_).union(set(self.t_eom))
        all_dates = list(all_dates)
        all_dates.sort()
        df = pd.DataFrame(self.t_eom, index=self.t_eom, columns=['eom']).reindex(all_dates)
        df.fillna(method='ffill', inplace=True)
        rtn = df.loc[date_list_]
        return rtn['eom'].tolist()

    def get_last_trading_eohs(self, date_list_: list, inc_self_if_is_trdday):
        assert inc_self_if_is_trdday
        all_dates = set(date_list_).union(set(self.t_eoh))
        all_dates = sorted(list(all_dates))
        df = pd.DataFrame(self.t_eoh, index=self.t_eoh, columns=['eoh']).reindex(all_dates)
        df.fillna(method='ffill', inplace=True)
        rtn = df.loc[date_list_]
        return rtn['eoh'].tolist()

    def get_n_years_before(self, date_list, n_):
        assert isinstance(date_list, list)
        rtn = (pd.to_datetime(date_list) - pd.Timedelta(days=365 * n_)).strftime('%Y-%m-%d').tolist()
        return rtn

    def get_latest_n_eow_trading_dates(self, date_: str, n_: int):
        dates = [x for x in self.t_eow if x <= date_][-n_:]
        return dates

    def get_latest_n_eom_trading_dates(self, date_: str, n_: int):
        dates = [x for x in self.t_eom if x <= date_][-n_:]
        return dates

    def get_latest_n_month_dates(self, date_: str, n_: int):
        assert n_ != 0
        y = int(date_[:4])
        m = int(date_[5:7])
        d = int(date_[8:])
        if n_ > 0:
            n = n_
            for i in range(n):
                if m == 1:
                    y -= 1
                    m = 12
                else:
                    m -= 1
        else:
            n = -n_
            for i in range(n):
                if m == 12:
                    y += 1
                    m = 1
                else:
                    m += 1
        y = str(y)
        m = format(m, "02d")
        d = format(d, "02d")
        assert y in self.ymd_dict and m in self.ymd_dict[y] and len(self.ymd_dict[y][m]) > 0
        last_d = self.ymd_dict[y][m][-1]
        if int(d) > int(last_d):
            d = last_d
        rtn = y + '-' + m + '-' + d
        return rtn

    def is_it(self, type_: str, date_: str):
        if type_.lower() == "teom":
            rtn = date_ in self.t_eom
        elif type_.lower() == "teow":
            rtn = date_ in self.t_eow
        elif type_.lower() == "is_trading":
            rtn = date_ in self.t_calendar
        else:
            raise NotImplementedError
        return rtn

    def count_trading_dates(self, start_date_, end_date_):
        assert isinstance(end_date_, (str, list))
        n_start_date = self.index_dict[start_date_]
        if isinstance(end_date_, str):
            n_end_date = self.index_dict[end_date_]
            assert n_start_date <= n_end_date
            rtn = n_end_date - n_start_date
        else:
            rtn = [self.index_dict[ed] - n_start_date for ed in end_date_]
        return rtn

    def count_dates(self, start_date_, end_date_):
        n_start_date = self.calendar_date_index_dict[start_date_]
        n_end_date = self.calendar_date_index_dict[end_date_]
        assert n_start_date is not None and n_end_date is not None
        return n_end_date - n_start_date

    def n_dates_from_anchor_date(self, date_):
        rtn = self.calendar_date_index_dict[date_]
        return rtn

    def get_ranged_eom_dates(self, start_date, end_date):
        return pd.date_range(start_date, end_date, freq='M').strftime('%Y-%m-%d').tolist()

    def get_ranged_eom_trading_dates(self, start_date_: str, end_date_: str):
        dates = [x for x in self.t_eom if (x >= start_date_) and (x <= end_date_)]
        return dates

    def get_ranged_eow_trading_dates(self, start_date: str, end_date: str):
        dates = [x for x in self.t_eow if start_date <= x <= end_date]
        return dates

    def get_ranged_eoh_trading_dates(self, start_date: str, end_date: str):
        dates = [x for x in self.t_eoh if start_date <= x <= end_date]
        return dates


CALENDAR_UTIL = CalendarUtil()