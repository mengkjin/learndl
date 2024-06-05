from basic_src_data.wind_tools.basic import get_n_listed_dates_from_winddf


def get_n_listed_dates_filter(start_date_, end_date_):
    rtn = get_n_listed_dates_from_winddf(start_date_, end_date_)
    return rtn