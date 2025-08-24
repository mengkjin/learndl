import os
from basic_src_data.wind_tools.divnsplit import load_divnsplit_winddf


def _save(root_path, df):
    path = os.path.sep.join([root_path, 'common_data', 'divnsplit'])
    if not os.path.exists(path):
        os.makedirs(path)
    df.sort_values(by=['ExDate', 'Code'], inplace=True)
    file_path = os.path.join(path, 'divnsplit.ftr')
    df.reset_index(drop=True, inplace=True)
    df.to_feather(file_path)


def gen_divnsplit_data(root_path, eed):
    sed = "2000-01-10"
    df = load_divnsplit_winddf(sed, eed)
    df = df.sort_values(["ExDate", "Code", "RecDate"]).groupby(by=["ExDate", "Code"], as_index=False).agg(
        {"RecDate": "last", "div_rate": "sum", "split_ratio": "sum"})
    _save(root_path, df)
    print("  status::divnsplit>>generator>>calculated and saved from {0} to {1} in path {2}.".format(
        sed, eed, root_path))