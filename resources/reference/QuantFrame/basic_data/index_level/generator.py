import pandas as pd
from basic_src_data.wind_tools.index import get_index_level_by_wind
from crosec_mem.data_vendor import save
from .configs import get_data_path


def gen_index_level_data(root_path, category, scd, ecd):
    if category == 'broad_based':
        index_list = ['000300.SH', '000905.SH', '000852.SH', '932000.CSI', '000906.SH', '931865.CSI']
        df = list()
        for index_nm in index_list:
            index_df = get_index_level_by_wind(scd, ecd, index_nm)
            index_df["index_code"] = index_nm
            df.append(index_df)
        df = pd.concat(df, axis=0, sort=True).sort_values(["CalcDate", "index_code"])
    else:
        assert False, "  error::index_level>>generator>>unknown category:{0}.".format(category)
    path = get_data_path(root_path, category)
    save(path, df, asset_col="index_code")
    print("  status::index_level>>generator>>calculated and saved from {0} to {1} for category {2} in path {3}.".format(
        scd, ecd, category, root_path))