from .configs import get_path
from crosec_mem.data_vendor import save, get_to_update_date_range
from .impl.citics import gen_citics_data


def calc_indicator(category, scd, ecd):
    if category == "citics":
        rtn = gen_citics_data(scd, ecd)
    else:
        assert False
    return rtn


def gen_industry_data(root_path, scd, ecd, category):
    path = get_path(root_path, category)
    to_update_date_range = get_to_update_date_range(path, scd, ecd)
    if to_update_date_range is not None:
        to_update_scd, to_update_ecd = to_update_date_range
        df = calc_indicator(category, to_update_scd, to_update_ecd)
        save(path, df, store_type="ftr")
        print("  status::industry>>generator>>calculated and saved from {0} to {1} for category {2} in path {3}.".format(
            scd, ecd, category, root_path))
    else:
        print("  status::industry>>generator>>data {0} already exists from {1} to {2}.".format(category, scd, ecd))
