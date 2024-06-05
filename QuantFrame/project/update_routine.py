from barra_model.factor_impl.generator import gen_barra_data
from daily_bar.generator import gen_daily_bar_data
from industry.generator import gen_industry_data
from ashare_stkpool.generator import gen_stkpool_data
from barra_model.risk_ret_est.generator import gen_risk_ret
from index_weight.generator import gen_index_weight_data
from index_level.generator import gen_index_level_data
from barra_model.model_impl.generator import gen_risk_cov, gen_special_vol
from divnsplit.generator import gen_divnsplit_data
from events_system.calendar_util import CALENDAR_UTIL
from stk_basic_info.generator import gen_stk_basic_info_data

# 注1：使用者需修改packages>>basic_src_data>>wind_tools>>wind_conn中的连接函数
# 注2：barra、股票池与基础数据之间有依赖关系，故此处的部分数据日期需更早一些
root_path = 'D:/QuantData'  # 文件存储路径
scd = '2023-12-31' 
ecd = '2024-05-24'
risk_ret_sd = CALENDAR_UTIL.get_last_trading_dates([scd], n=505, inc_self_if_is_trdday=False)[0]  # 因子收益数据的起始日期
barra_sd = CALENDAR_UTIL.get_last_trading_dates([risk_ret_sd], n=2, inc_self_if_is_trdday=False)[0]  # barra因子数据的起始日期
dbar_sd = CALENDAR_UTIL.get_last_trading_dates([barra_sd], n=501, inc_self_if_is_trdday=False)[0]  # 日行情数据的起始日期
#
gen_stk_basic_info_data(root_path, "description")  # 生成股票基础信息数据
gen_stk_basic_info_data(root_path, "st")  # 生成st数据
#
gen_daily_bar_data(root_path, "basic", dbar_sd, ecd)  # 生成日行情数据
gen_daily_bar_data(root_path, "valuation", dbar_sd, ecd)  # 生成估值数据
gen_industry_data(root_path, dbar_sd, ecd, "citics")  # 生成中信行业数据
gen_index_weight_data(root_path, "broad_based", dbar_sd, ecd)  # 生成指数权重数据
gen_index_level_data(root_path, "broad_based", dbar_sd, ecd)  # 生成指数日行情数据
gen_divnsplit_data(root_path, ecd)  # 生成分红数据
#
# 生成barra因子数据，须有最近500个交易日的日行情与估值值
gen_barra_data(root_path, "cne6", barra_sd, ecd)
# 生成股票池数据，须有当天的barra因子值
gen_stkpool_data(root_path, barra_sd, ecd)
# 生成barra因子收益与股票特质收益数据，须有前一天的barra因子值
gen_risk_ret(root_path, "cne6", risk_ret_sd, ecd)
# 生成barra特质波动率数据，须有最近252个交易日的股票特质收益数据
gen_special_vol(root_path, "cne6", scd, ecd)
# 生成barra风险因子协方差矩阵数据，须有最近504个交易日的因子收益数据
gen_risk_cov(root_path, "cne6", scd, ecd)