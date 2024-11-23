import os
import yaml
from factor_perf_tests.api import test_factor, agg_factor
from events_system.samplings import resample_trd_calendar_by_dates
from factor_loader_utils.factor_loader import load_factor_data
from factor_tools.winsorizor import winsorize
from factor_tools.normalizer import normalize
from factor_tools.neutralizer import neutralize
from barra_model.factor_impl.api import merge_with_barra_data

# 设置日期和因子
scd = "2020-01-01"
ecd = "2024-03-29"
freq_type = "week"  # 因子频率
root_path = "D:/QuantData"
factor_path = "D:/QuantData/factor_data"  # 因子存储路径
factor_name = "analyst_l"  # 因子名称

# 设置参数
config_path = os.path.abspath("tf_config.yaml")
with open(config_path, "rb") as file:
    configs = yaml.load(file.read(), Loader=yaml.SafeLoader)

# 读取因子数据
date_list = resample_trd_calendar_by_dates(scd, ecd, freq_type, smp_type="end")
factor_val_df = load_factor_data(factor_path, factor_name, date_list)
factor_val_df = factor_val_df.rename(columns={factor_name: factor_name + "-raw"}, errors="raise")

# 因子处理
if configs["PROC_CONFIG"]["winsorize"]:
    factor_val_df = winsorize(factor_val_df, date_code_cols=["CalcDate", "Code"])
if configs["PROC_CONFIG"]["neutralize"]:
    factor_val_df = merge_with_barra_data(
        root_path, factor_val_df, configs["CALC_CONFIG"]["barra_type"], configs["PROC_CONFIG"]["style_list"])
    factor_val_df = neutralize(
        factor_val_df, numeric_x_cols=configs["PROC_CONFIG"]["style_list"],
        categorical_x_col=configs["PROC_CONFIG"]["industry_fld"],
        prefix="", date_code_cols=["CalcDate", "Code"])[0]
    factor_val_df = factor_val_df.rename(columns={factor_name + "-raw": factor_name + "-neut"}, errors="raise")
if configs["PROC_CONFIG"]["normalize"]:
    factor_val_df = normalize(factor_val_df, date_code_cols=["CalcDate", "Code"])

#
input_dict = {
    "ret_end_date": ecd,
    "test_scope": ["all"],  # 测试基于的行业，all表示全市场
    "to_agg": "on",  # 'on' / 'off'，agg操作会将因子结果另行整合到agg_factor_results_tf文件中，方便比较多个因子
    "agg_result_save_path": "D:/QuantData/factor_results",
    "agg_file_name": "agg_factor_results_tf",
}

# 因子测试
factor_list = factor_val_df.columns.drop(["CalcDate", "Code"]).tolist()
for factor in factor_list:
    factor_val = factor_val_df[["CalcDate", "Code", factor]].dropna(subset=[factor])
    test_rslt_excel = test_factor(root_path, factor_val, freq_type, ecd, input_dict["test_scope"], configs)
    if input_dict["to_agg"] == "on":
        agg_factor(test_rslt_excel, input_dict["agg_result_save_path"], input_dict["agg_file_name"])