from basic_src_data.wind_tools.valuation import load_valuation_by_wind


def get_bookvalue_filter(start_date_, end_date_):
    rtn = load_valuation_by_wind(start_date_, end_date_, ['book_value'])
    return rtn



