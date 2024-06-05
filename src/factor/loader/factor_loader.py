import os
import pandas as pd


def _add_suffix_to_code(code):
    first_num = code[0]
    if first_num == '6':
        rtn = code + '.SH'
    elif first_num == '0' or first_num == '3':
        rtn = code + '.SZ'
    elif first_num == '8':
        rtn = code + '.BJ'
    else:
        assert False
    return rtn


def transform_data_code_type(daily_data):
    daily_data = daily_data[daily_data["Code"].str[0].isin(["0", "3", "6", "8"])].copy()
    daily_data["Code"] = daily_data["Code"].str[:6].apply(_add_suffix_to_code)
    return daily_data


def load_factor_data(factor_root_path, factor_name, date_list):
    path = os.path.join(factor_root_path, factor_name)
    assert os.path.exists(path), "  error::>>factor_loader>>Cannot find path {0}".format(path)
    file_list = os.listdir(path)
    #
    file_scope = ['{0}_{1}.txt'.format(factor_name, d.replace("-", "")) for d in date_list]
    file_list_missing = set(file_scope).difference(set(file_list))
#    assert not file_list_missing, "  error::>>factor_loader>> Could not find file {0}!".\
#        format(",".join(list(file_list_missing)))
    if file_list_missing:
        print("  warning::>>factor_loader>> Could not find file {0}!".format(",".join(list(file_list_missing))))
    file_list_chosen = set(file_scope) & set(file_list)
    rtn = []
    for file_name in file_list_chosen:
        date = os.path.splitext(file_name)[0].split("_")[-1]
        date = "-".join([date[:4], date[4:6], date[6:8]])
        if date in date_list:
            file_path = os.path.join(path, file_name)
            day_factor = pd.read_table(file_path, header=None)
            day_factor.columns = ["Code", factor_name]
            day_factor["Code"] = day_factor["Code"].astype(str).str.rjust(6, "0")
            day_factor["CalcDate"] = date
            day_factor = transform_data_code_type(day_factor)
            rtn.append(day_factor)
    assert len(rtn) > 0, "  error::>>factor_loader>>There is no corresponding data for the selected time interval"
    rtn = pd.concat(rtn, axis=0)
    assert (rtn.shape[1] == 3) & ({"CalcDate", "Code"}.issubset(set(rtn.columns))), \
        "  error::factor_loader>>Incorrect data column name"
    if not rtn.notna().all().all():
        rtn = rtn.dropna()
        print("  warning::>>factor_loader>>factor {0} has nan value, drop it".format(factor_name))
    rtn = rtn[["CalcDate", "Code"] + rtn.columns.drop(["CalcDate", "Code"]).tolist()
              ].sort_values(["CalcDate", "Code"]).reset_index(drop=True)
    return rtn