import pandas as pd
import os
from events_system.calendar_util import CALENDAR_UTIL


def load_portfolio_data(port_path, port_name, date_list):
    path = os.path.join(port_path, port_name)
    assert os.path.exists(path), " error:>>port_loader>>Cannot find path {0}".format(
        path)
    file_list = os.listdir(path)
    #
    file_scope = ['{0}_{1}.xlsx'.format(port_name, d.replace("-", "")) for d in date_list]
    file_list_missing = set(file_scope).difference(set(file_list))
    assert not file_list_missing, "  error:>>port_loader>> Could not find file {0}!". \
        format(",".join(list(file_list_missing)))
    file_list_chosen = set(file_scope) & set(file_list)
    rtn = []
    for file_name in file_list_chosen:
        date = os.path.splitext(file_name)[0].split("_")[-1]
        date = "-".join([date[:4], date[4:6], date[6:8]])
        if date in date_list:
            file_path = os.path.join(path, file_name)
            day_factor = pd.read_excel(file_path, sheet_name=port_name, header=1)
            day_factor = day_factor[["代码", "占组合净值"]].rename(
                columns={"代码": "Code", "占组合净值": "target_weight"}, errors="raise")
            day_factor["TradeDate"] = date
            rtn.append(day_factor)
    assert len(rtn) > 0, "  error::>>port_loader>>There is no corresponding data for the selected time interval"
    rtn = pd.concat(rtn, axis=0)
    assert (rtn.shape[1] == 3) & ({"TradeDate", "Code"}.issubset(set(rtn.columns))), \
        "  error::port_loader>>Incorrect data column name"
    if not rtn.notna().all().all():
        rtn = rtn.dropna()
        print("  warning::>>port_loader>>portfolio {0} has nan value, drop it".format(port_name))
    rtn["Code"] = rtn["Code"].str.replace("SS", "SH")
    #
    rtn = rtn.groupby(["TradeDate", "Code"])["target_weight"].sum().reset_index(drop=False)
    rtn["CalcDate"] = CALENDAR_UTIL.get_last_trading_dates(rtn["TradeDate"].tolist(), inc_self_if_is_trdday=False)
    rtn = rtn[["TradeDate", "CalcDate", "Code", "target_weight"]].sort_values(["TradeDate", "Code"]).reset_index(drop=True)
    return rtn