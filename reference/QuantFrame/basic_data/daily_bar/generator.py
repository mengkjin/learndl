from .configs import *
from daily_bar.center import calc_basic_data, calc_valuation_data
from crosec_mem.data_vendor import save, get_to_update_date_range


def gen_daily_bar_data(root_path, category, scd, ecd):
    path = get_data_path(root_path, category)
    date_type_dict = {"basic": "trade", "valuation": "full"}
    to_update_date_range = get_to_update_date_range(path, scd, ecd, date_type_dict[category])
    if to_update_date_range is not None:
        to_update_scd, to_update_ecd = to_update_date_range
        if category == 'basic':
            df = calc_basic_data(to_update_scd, to_update_ecd)
        elif category == 'valuation':
            df = calc_valuation_data(to_update_scd, to_update_ecd)
        else:
            assert False, "  error::daily_bar>>generator>>unknown category:{0}.".format(category)
        save(path, df)
        print("  status::daily_bar>>generator>>calculated and saved from {0} to {1} for category {2} in path {3}.".format(
            to_update_scd, to_update_ecd, category, root_path))
    else:
        print("  status::daily_bar>>generator>>data {0} already exists from {1} to {2}.".format(category, scd, ecd))
