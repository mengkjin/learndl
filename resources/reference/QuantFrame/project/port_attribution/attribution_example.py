import os
import yaml
from fmp_perf_tests.api import analyse_port_perf
from events_system.samplings import resample_trd_calendar_by_dates
from port_generals.port_loader import load_portfolio_data
from ashare_stkpool.api import remove_outpool_data

# 设置日期和组合
scd = "2023-09-01"
ecd = "2024-03-08"
root_path = "D:/QuantData"
port_path = "D:/QuantData/port_data"  # 组合存储路径
#port_path = "//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Portfolio"  # 组合存储路径"
port_name = "社保903"  # 组合名称
freq_type = "week"  # 组合频率
bm_index = "publish:000906.SH"  # 基准指数

# 设置参数
config_path = os.path.join(os.path.dirname(__file__), "attribution_config.yaml")
with open(config_path, "rb") as file:
    test_configs = yaml.safe_load(file.read())
test_configs["ENV_CONFIG"] = {"stock_universe": "universe", "risk_model_nm": "cne6"}
test_configs["BACKTEST_CONFIG"]["bm_index"] = bm_index

# 读取组合数据
date_list = resample_trd_calendar_by_dates(scd, ecd, freq_type, smp_type="end")
port_weight = load_portfolio_data(port_path, port_name, date_list)

#
# 去除组合中未上市股票及退市股票，避免程序报错
port_weight = remove_outpool_data(root_path, port_weight, pool_type="ashares")
# 归因时是否需要考虑仓位带来的贡献，考虑仓位时需要去掉下面的权重归一化操作
port_weight["target_weight"] = port_weight["target_weight"] / port_weight.groupby(["CalcDate"])["target_weight"].transform("sum")
# 对组合进行回测并分析收益（注：费率等参数不适用于收益分解的结果）
analyse_port_perf(root_path, port_weight, ecd, test_configs, port_name, freq_type)