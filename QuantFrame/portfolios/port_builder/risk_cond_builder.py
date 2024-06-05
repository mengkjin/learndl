import numpy as np
import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from barra_model.model_impl.api import load_special_vol, load_risk_cov
from barra_model.factor_impl.api import load_barra_data
from ashare_stkpool.api import load_stkpool_data_by_dates
from daily_bar.api import load_daily_bar_data
from stk_index_utils.api import load_index_weight_data
from factor_tools.general import onehotize


def build_linear_risk_conditions(conditions, industry_bias, style_bias, bm_least_weight, leverage):
    conditions = conditions.copy()
    bm_risks = conditions[industry_bias.index.append(style_bias.index)].T.dot(conditions['rsk_bm'])
    ind_target_risks = bm_risks.loc[industry_bias.index].rename('bm').to_frame()
    ind_target_risks['boundkey'] = 'ra'
    ind_target_risks[['bias_ub', 'bias_lb']] = industry_bias[['bias_ub', 'bias_lb']]
    ind_target_risks['ub'] = ind_target_risks['bm'] + ind_target_risks['bias_ub']
    ind_target_risks['lb'] = ind_target_risks['bm'] + ind_target_risks['bias_lb']
    ind_target_risks['lb'] = ind_target_risks['lb'].where(ind_target_risks['lb'] > 0.0, 0.0)
    # ind_target_risks.loc[:, ['ub', 'lb']] = pd.concat((
    #     (bm_risks.loc[industry_bias.index] + industry_bias['bias_ub']).rename('ub'),
    #     (bm_risks.loc[industry_bias.index] + industry_bias['bias_lb']).rename('lb')), axis=1)
    ind_target_risks = ind_target_risks[['ub', 'lb', 'boundkey']].copy()
    #
    style_target_risks = bm_risks.loc[style_bias.index].rename('bm').to_frame()
    style_target_risks[['bias_ub', 'bias_lb']] = style_bias[['bias_ub', 'bias_lb']]
    style_target_risks['ub'] = style_target_risks['bm'] + style_target_risks['bias_ub']
    style_target_risks['lb'] = style_target_risks['bm'] + style_target_risks['bias_lb']
    style_target_risks['boundkey'] = 'ra'
    style_target_risks = style_target_risks[['ub', 'lb', 'boundkey']].copy()
    #
    target_risks = pd.concat((ind_target_risks, style_target_risks), axis=0)
    #
    risk_factors = conditions[target_risks.index].T
    risk_target = target_risks.copy()
    # index member
    assert len(bm_least_weight) == 1
    bm_nm, bmlw_val = bm_least_weight.index[0], bm_least_weight.values[0]
    risk_factors = pd.concat((risk_factors, ((conditions[bm_nm] > 1e-5) * 1.0).to_frame().T), axis=0)
    risk_target = pd.concat((risk_target, pd.Series((1.0, bmlw_val, 'lo'), index=['ub', 'lb', 'boundkey'], name=bm_nm).to_frame().T), axis=0)
    # risk_factors = risk_factors.append((conditions[bm_nm] > 1e-5) * 1.0)
    # risk_target = risk_target.append(pd.Series((1.0, bmlw_val, 'lo'), index=['ub', 'lb', 'boundkey'], name=bm_nm))
    # leverage
    risk_factors = pd.concat(
        (risk_factors, pd.Series(np.ones(len(conditions)), name='leverage', index=conditions.index).to_frame().T), axis=0)
    risk_target = pd.concat(
        (risk_target, pd.Series((leverage, leverage, 'fx'), index=['ub', 'lb', 'boundkey'], name='leverage').to_frame().T),
        axis=0)
    # risk_factors = risk_factors.append((pd.Series(np.ones(len(conditions)), name='leverage', index=conditions.index)))
    # risk_target = risk_target.append(pd.Series((1.0, 1.0, 'fx'), index=['ub', 'lb', 'boundkey'], name='leverage'))
    assert risk_factors.notna().all().all() and risk_target.notna().all().all()
    return risk_factors, risk_target


# def build_bound_conditions(conditions, upper_bound, lower_bound, abs_or_exc='abs'):
def build_bound_conditions(conditions, abs_bnds, exc_bnds, no_sell_codes, no_buy_codes):
    conditions = conditions.set_index(['Code'])
    # abs
    abs_ub = pd.Series([abs_bnds[1]] * len(conditions), index=conditions.index)
    abs_lb = pd.Series([abs_bnds[0]] * len(conditions), index=conditions.index)
    abs_bnd_key = pd.Series(['ra'] * len(conditions), index=conditions.index)
    abs_bnd_df = pd.concat((abs_bnd_key.rename('boundkey'), abs_lb.rename('lb'), abs_ub.rename('ub')), axis=1)
    # exc
    bm_weight = conditions['bm_index'].copy()
    exc_ub = bm_weight + exc_bnds[1]
    exc_lb = bm_weight + exc_bnds[0]
    exc_lb[exc_lb < 0.0] = 0.0
    exc_bnd_key = pd.Series(['ra'] * len(conditions), index=conditions.index)
    exc_bnd_df = pd.concat((exc_bnd_key.rename('boundkey'), exc_lb.rename('lb'), exc_ub.rename('ub')), axis=1)
    #
    # combine
    ub = pd.concat((abs_bnd_df["ub"], exc_bnd_df['ub']), axis=1).min(axis=1)
    lb = pd.concat((abs_bnd_df["lb"], exc_bnd_df['lb']), axis=1).max(axis=1)
    #
    w0 = conditions["w0"].copy()
    lb[no_sell_codes] = w0[no_sell_codes]
    ub[no_buy_codes] = w0[no_buy_codes]
    #
    assert (ub >= lb).all()
    bnd_key = abs_bnd_df["boundkey"]
    rtn = pd.concat((bnd_key.rename('boundkey'), lb.rename('lb'), ub.rename('ub')), axis=1)
    return rtn


# def build_bound_conditions1(conditions, upper_bound, lower_bound, abs_or_exc='abs', bm_index=None):
#     conditions = conditions.set_index(['Code'])
#     if abs_or_exc == 'abs':
#         ub = pd.Series([upper_bound] * len(conditions), index=conditions.index)
#         lb = pd.Series([0.0] * len(conditions), index=conditions.index)
#         bnd_key = pd.Series(['ra'] * len(conditions), index=conditions.index)
#         blk_key = conditions['is_black'] > 0
#         bm_key = conditions['bm_flg'] > 0.5
#         bad_bm_flg = blk_key & bm_key
#         ub[bad_bm_flg] = conditions.loc[bad_bm_flg, 'bm_index']
#         bad_nonbm_flg = blk_key & (~bm_key)
#         ub[bad_nonbm_flg] = 0.0
#         bnd_key[bad_nonbm_flg] = 'fx'
#     elif abs_or_exc == 'exc':
#         bm_weight = conditions['bm_index'].copy()
#         ub = bm_weight + upper_bound
#         lb = bm_weight + lower_bound
#         lb[lb < 0.0] = 0.0
#         bnd_key = pd.Series(['ra'] * len(conditions), index=conditions.index)
#     else:
#         assert False
#     return pd.concat((bnd_key.rename('bound_key'), lb.rename('lb'), ub.rename('ub')), axis=1)


def build_quad_risk_conditions(conditions, excess_var_ratio, bm_least_weight):  # bm_least_weight tmp
    # TMP START
    bm_nm = bm_least_weight.index[0]
    # TMP END
    conditions = conditions.copy()
    special_var = conditions['svol'] * conditions['svol']
    bm_var = np.sum(conditions[bm_nm] * conditions[bm_nm] * special_var)
    diag_part = special_var.copy()
    linear_part = - conditions[bm_nm] * special_var
    bnds = (0.5 * bm_var * (excess_var_ratio - 1.0), 0.0, 'up')

    rtn_linear = linear_part.rename('0').to_frame().T
    rtn_bnds = pd.DataFrame([bnds], index=['0'], columns=['ub', 'lb', 'boundkey'])
    return diag_part, rtn_linear, rtn_bnds


def get_general_conditions(root_path, stk_pool_type, barra_type, index_nm, port_calc_date_list):
    latest_traded_date_list = CALENDAR_UTIL.get_last_trading_dates(port_calc_date_list, True)
    conditions = pd.DataFrame(zip(port_calc_date_list, latest_traded_date_list), columns=['port_calc_date', 'latest_trade_date'])
    stk_pool = load_stkpool_data_by_dates(root_path, port_calc_date_list, stk_pool_type)
    conditions = pd.merge(conditions, stk_pool, how='left', left_on=['port_calc_date'], right_on=['CalcDate']).drop(columns=['CalcDate'])
    barra_data = load_barra_data(root_path, barra_type, port_calc_date_list[0], port_calc_date_list[-1])
    barra_ind_col_list = barra_data.columns[barra_data.columns.str.contains('INDUSTRY\.')].copy()
    assert len(barra_ind_col_list) == 1
    barra_ind_col = barra_ind_col_list[0]
    conditions = pd.merge(conditions, barra_data, how='left', left_on=['port_calc_date', 'Code'], right_on=['CalcDate', 'Code']).drop(columns=['CalcDate'])
    is_traded = load_daily_bar_data(root_path, 'basic', latest_traded_date_list[0], latest_traded_date_list[-1])[['CalcDate', 'Code', 'is_traded']]
    conditions = pd.merge(conditions, is_traded, how='inner', left_on=['latest_trade_date', 'Code'], right_on=['CalcDate', 'Code']).drop(columns=['CalcDate'])
    bm_weights = load_index_weight_data(root_path, latest_traded_date_list[0], latest_traded_date_list[-1], index_nm)
    conditions = pd.merge(conditions, bm_weights, how='left', left_on=['latest_trade_date', 'Code'], right_on=['CalcDate', 'Code']).drop(
        columns=['CalcDate'])
    conditions["member_weight"].fillna(0.0, inplace=True)
    special_vols = load_special_vol(root_path, barra_type, port_calc_date_list[0], port_calc_date_list[-1]).rename(
        columns={"special_vol": "svol"}, errors="raise")
    conditions = pd.merge(conditions, special_vols, how='inner', left_on=['port_calc_date', 'Code'], right_on=['CalcDate', 'Code']).drop(
        columns=['CalcDate'])
    onehotized_industry = onehotize(conditions[[barra_ind_col]])
    onehotized_industry.columns = ['INDUSTRY.{0}'.format(c) for c in onehotized_industry.columns]
    conditions = pd.concat((conditions, onehotized_industry), axis=1)
    conditions.rename(columns={barra_ind_col: "INDUSTRY.Name"}, errors="raise", inplace=True)
    conditions.dropna(how='any', inplace=True)
    conditions.drop(columns=['latest_trade_date'], inplace=True)
    conditions.rename(columns={'port_calc_date': 'CalcDate'}, errors='raise', inplace=True)
    assert conditions['CalcDate'].drop_duplicates().tolist() == port_calc_date_list
    return conditions


def get_cov_mat(root_path, barra_type, port_calc_date_list):
    risk_cov = load_risk_cov(root_path, barra_type, port_calc_date_list[0], port_calc_date_list[-1])
    risk_cov = risk_cov[risk_cov["CalcDate"].isin(port_calc_date_list)].copy()
    rtn = dict()
    for date in port_calc_date_list:
        cov_data = risk_cov[risk_cov["CalcDate"] == date].copy()
        cov_data = cov_data.drop(columns=["CalcDate"]).set_index("FactorName")
        rtn[date] = cov_data
    return rtn

