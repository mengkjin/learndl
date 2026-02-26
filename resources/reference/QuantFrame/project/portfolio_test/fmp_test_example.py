import os
import yaml
from fmp_perf_tests.api import test_fmp_perf, agg_fmp_perf
from events_system.samplings import resample_trd_calendar_by_dates
from factor_loader_utils.factor_loader import load_factor_data
from barra_model.risk_ret_est.api import get_riskmodel_lastest_days

# 设置参数
config_path = os.path.join(os.path.dirname(__file__), 'fmp_config_csi800.yaml')
with open(config_path, "rb") as file:
    configs = yaml.safe_load(file.read())

# 设置日期和因子
root_path = "D:/QuantData"
scd, ecd = get_riskmodel_lastest_days(root_path, configs["ENV_CONFIG"]["risk_model_nm"], 252)
freq_type = "week"  # 因子频率
factor_path = "D:/QuantData/factor_data"  # 因子数据的本地路径
factor_name = "analyst_l"  # 因子名称，需与本地路径名称对应

# 
date_list = resample_trd_calendar_by_dates(scd, ecd, freq_type, smp_type="end")  # 按照频率获取因子
fcst_df = load_factor_data(factor_path, factor_name, date_list)  # 读取因子数据

#
xlsx_path = test_fmp_perf(root_path, fcst_df, ecd, configs, freq_type)  # 使用组合优化计算组合权重并进行回测
#agg_fmp_perf(xlsx_path, configs["SAVE_CONFIG"]["agg_path"], "agg_fmp_perf")  # 将测试结果存在agg_fmp_perf文件中，用于比较多次测试结果

'''
#
# 若只需要组合优化工具计算权重，可使用以下代码：
from port_builder.seq_portcons import calc_target_weights
fcst_data = fcst_df.rename(columns={factor_name: "fcst"}, errors="raise")
target_weight = calc_target_weights(root_path, fcst_data, {"freq": "adapt_to_fcst"}, configs["ENV_CONFIG"],
                                    {"portfolio": configs["OPT_CONFIG"]},
                                    fcst_data["CalcDate"].iloc[0], fcst_data["CalcDate"].iloc[-1])["portfolio"]
print(target_weight)
#target_weight.to_csv("fmp_test_output.csv")
'''