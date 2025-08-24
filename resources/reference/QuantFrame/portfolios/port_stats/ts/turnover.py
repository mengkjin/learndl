import pandas as pd
from port_generals.weight_utils import calc_port_eod_weight


def calc_port_expected_turnover(root_path, port_weight):
    assert pd.Index(["CalcDate", "TradeDate", "Code", "target_weight"]).difference(port_weight.columns).empty
    assert port_weight[["CalcDate", "TradeDate", "Code", "target_weight"]].notna().all().all()
    port_weight = port_weight.sort_values(["CalcDate", "Code"])
    port_weight_filled = calc_port_eod_weight(root_path, port_weight, port_weight["CalcDate"].max())
    port_weight_filled = port_weight_filled.loc[port_weight_filled["weight"].abs() > 1e-5,
                                                ["WeightDate", "Code", "weight"]].copy()
    #
    calc_to_dates = port_weight["CalcDate"].unique().tolist()[1:]
    port_weight_filled = port_weight_filled[port_weight_filled["WeightDate"].isin(calc_to_dates)].copy()
    all_data = pd.merge(port_weight[port_weight["CalcDate"].isin(calc_to_dates)],
                        port_weight_filled.rename(columns={"WeightDate": "CalcDate"}, errors="raise"),
                        on=["CalcDate", "Code"], how="outer")
    all_data["delta_weight_abs"] = (all_data["target_weight"].fillna(0.0) - all_data["weight"].fillna(0.0)).abs()
    rtn = all_data.groupby(["CalcDate"])["delta_weight_abs"].sum() / 2
    rtn = rtn.rename("expected_turnover").reset_index(drop=False)
    return rtn


def calc_port_expected_turnover_by_filled_weight(port_weight):
    assert pd.Index(["WeightDate", "Code", "weight", "expected_weight"]).difference(port_weight.columns).empty
    assert port_weight[["WeightDate", "Code", "weight", "expected_weight"]].notna().all().all()
    all_data = port_weight.copy()
    all_data["delta_weight_abs"] = (all_data["expected_weight"] - all_data["weight"]).abs()
    rtn = all_data.groupby(["WeightDate"])["delta_weight_abs"].sum() / 2
    rtn = rtn.rename("expected_turnover").reset_index(drop=False).rename(columns={"WeightDate": "CalcDate"})
    return rtn