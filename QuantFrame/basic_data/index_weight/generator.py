from crosec_mem.data_vendor import save, get_to_update_date_range
import pandas as pd
from basic_src_data.wind_tools.index import get_index_member_weight_by_cumprod, get_common_index_member_weight
from common_tools.logger import write_log
from .configs import get_data_path


def gen_index_weight_data(root_path, category, scd, ecd):
    path = get_data_path(root_path, category)
    to_update_date_range = get_to_update_date_range(path, scd, ecd, "trade")
    if to_update_date_range is not None:
        to_update_scd, to_update_ecd = to_update_date_range
        if category == 'broad_based':
            index_list = ['000300.SH', '000905.SH', '000852.SH', '932000.CSI', '000906.SH', '931865.CSI']
            common_index = get_common_index_member_weight(to_update_scd, to_update_ecd)
            config_flg = (common_index[['000300.SH', '000905.SH', '000852.SH', '932000.CSI']] > 0.0).sum(axis=1) > 1
            if config_flg.any():
                write_log("index_weight.log", "data error: {0}".format(common_index[config_flg]))
                assert False, "  error::index_weight>>generator>>index weight conflict"
            #
            other_index_nms = list(set(index_list) - {'000300.SH', '000905.SH', '000852.SH', '932000.CSI'})
            df = common_index.copy()
            for index_nm in other_index_nms:
                new_index = get_index_member_weight_by_cumprod(to_update_scd, to_update_ecd, index_nm)
                df = pd.merge(df, new_index, how="outer", on=["CalcDate", "Code"], sort=True)
            df.fillna(0.0, inplace=True)
            df.reset_index(drop=False, inplace=True)
        else:
            assert False, "  error::index_weight>>generator>>unknown category:{0}.".format(category)
        save(path, df)
        print("  status::index_weight>>generator>>calculated and saved from {0} to {1} for category {2} in path {3}.".format(
            to_update_scd, to_update_ecd, category, root_path))
    else:
        print("  status::index_weight>>generator>>data {0} already exists from {1} to {2}.".format(category, scd, ecd))