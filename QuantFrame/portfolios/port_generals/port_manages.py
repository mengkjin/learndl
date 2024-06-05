import pandas as pd
import os


def get_port_file(port_path, port_cate):
    file_path = os.path.join(port_path, port_cate + '.xlsx')
    return file_path


def save_port(port_path, port_cate, port_dict):
    assert isinstance(port_dict, dict)
    if not os.path.exists(port_path):
        os.makedirs(port_path)
    file_path = get_port_file(port_path, port_cate)
    if os.path.exists(file_path):
        port_data = pd.read_excel(file_path, sheet_name=None)
    else:
        port_data = dict()
    for strategy, strategy_port in port_dict.items():
        assert not strategy_port.duplicated(["CalcDate", "Code"]).any()
        strategy_port.sort_values(by=['CalcDate', "Code"], inplace=True)
        if strategy in port_data.keys():
            local_strategy = port_data[strategy].copy()
            new_calc_dates = set(strategy_port["CalcDate"]).difference(set(local_strategy["CalcDate"]))
            strategy_port = pd.concat(
                (
                    local_strategy, strategy_port[strategy_port["CalcDate"].isin(new_calc_dates)]
                ), axis=0)
            strategy_port.sort_values(["CalcDate", "Code"], inplace=True)
        port_data[strategy] = strategy_port
    with pd.ExcelWriter(file_path) as writer:
        for strategy, strategy_port in port_data.items():
            strategy_port.to_excel(writer, sheet_name=strategy, index=False)


def load_port(port_path, port_cate):
    file_path = get_port_file(port_path, port_cate)
    assert os.path.exists(file_path), "  error:>>port_manages>>load_port_from_excel>>Can not find file in path {0}".format(file_path)
    rtn = pd.read_excel(file_path, sheet_name=None)
    assert isinstance(rtn, dict)
    for port_weight in rtn.values():
        assert pd.Index(["TradeDate", "CalcDate", "Code", "target_weight"]).difference(port_weight.columns).empty
        assert port_weight["TradeDate"].is_monotonic_increasing
    return rtn


def get_port_info(port_path, port_cate):
    file_path = get_port_file(port_path, port_cate)
    rtn = dict()
    if os.path.exists(file_path):
        port_data = pd.read_excel(file_path, sheet_name=None)
        for strategy in port_data.keys():
            strategy_port_df = port_data[strategy]
            rtn[strategy] = {
                'calc_dates': strategy_port_df['CalcDate'].drop_duplicates().tolist()}
    return rtn