import pandas as pd
from .configs import *
from crosec_mem.data_vendor import get_to_update_date_range, save


def gen_barra_data(root_path, barra_type, scd, ecd):
    path = get_data_path(root_path, barra_type)
    to_update_date_range = get_to_update_date_range(path, scd, ecd)
    if to_update_date_range is not None:
        to_update_scd, to_update_ecd = to_update_date_range
        folder_name = 'barra_model.factor_impl.{0}.center'.format(barra_type)
        mod = __import__(name=folder_name, fromlist=["calc_barra_vals"])
        calc_barra_vals = getattr(mod, "calc_barra_vals")
        df = calc_barra_vals(root_path, to_update_scd, to_update_ecd)
        save(path, df)
        print("  status::barra>>generator>>generate barra data from '{0}' to '{1} for type:{2}'.".format(to_update_scd, to_update_ecd, barra_type))
    else:
        print("  status::barra>>generator>>barra data {0} already exists from {1} to {2}.".format(barra_type, scd, ecd))