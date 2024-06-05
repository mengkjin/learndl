from optimizer.api import exec_socp
import numpy as np
import pandas as pd
from .risk_cond_builder import build_bound_conditions, build_linear_risk_conditions


def create_risk_bnds(risks, data_flds):
    industry_cols = data_flds[data_flds.str.contains('INDUSTRY\.')].drop(["INDUSTRY.Name"])
    ind_bias = pd.DataFrame([risks['industry']] * len(industry_cols), index=industry_cols, columns=["bias_lb", "bias_ub"])
    style_bias = pd.DataFrame(risks['style'], index=['bias_lb', 'bias_ub']).T
    #
    bm_least_weight = pd.Series([risks['bm_weight_bound']], index=['bm_index'])
    #
    rtn = {
            'ind': ind_bias,
            'style': style_bias,
            'bm_index': bm_least_weight,
            'exc_bounds': risks['exc_bounds'],
            'abs_bounds': risks['abs_bounds'],
            'te': risks['tracking_error'],
            'to': risks["turnover"],
            'leverage': risks["leverage"],
            'no_sell_codes': risks["no_sell_codes"],
            'no_buy_codes': risks["no_buy_codes"],
            }
    return rtn


def get_optim_param(conditions, port_con):
    risks = create_risk_bnds(port_con["RISK"], conditions.columns)
    linear_conditions = build_linear_risk_conditions(conditions, risks['ind'], risks['style'], risks['bm_index'], risks['leverage'])[1]
    linear_conditions.rename(index={"bm_index": "bm_flg"}, errors="raise", inplace=True)
    bnds = build_bound_conditions(conditions, risks['abs_bounds'], risks['exc_bounds'], risks['no_sell_codes'], risks['no_buy_codes'])
    #
    special_var = conditions['svol'] * conditions['svol']
    bm_var = np.sqrt(np.sum(conditions["bm_index"] * conditions["bm_index"] * special_var))
    #
    if risks["te"]["error_type"] == 'relative':
        te = bm_var * (risks["te"]["bound"] - 1.0)
    elif risks["te"]["error_type"] == 'absolute':
        te = risks["te"]["bound"]
    else:
        assert False
    #
    lmbd = port_con["LAMBDA"]
    rho = port_con["RHO"]
    to = (risks["leverage"] - conditions['w0'].sum()) + risks["to"]
    to_grad = port_con["RISK"]["turnover_gradient"]
    return lmbd, rho, te, to, linear_conditions, bnds, to_grad


def con_port(conditions, cov_mat, alf_col, port_con):
    conditions.reset_index(drop=True, inplace=True)
    conditions['COUNTRY'] = 1.0
    conditions['bm_flg'] = (conditions['bm_index'] > 0.000001) * 1.0
    conditions['bm_index'] = conditions['bm_index'] / conditions['bm_index'].sum()
    conditions["leverage"] = 1.0
    #
    lmbd, rho, te, to, rsk_bnds, bounds, to_grad = get_optim_param(conditions, port_con)
    risk_cols = conditions.columns[conditions.columns.str.contains('STYLE|INDUSTRY\.')].drop(["INDUSTRY.Name"])
    #
    F = conditions.loc[:, risk_cols].to_numpy().T
    u = conditions[alf_col].to_numpy()
    S = (conditions['svol'] ** 2).to_numpy()
    C = cov_mat.loc[risk_cols, risk_cols].to_numpy()
    wb = conditions['bm_index'].to_numpy()
    w0 = conditions['w0'].to_numpy()
    cov_info = (lmbd, F, C, S)
    lin_con = (conditions[rsk_bnds.index].to_numpy().T, [rsk_bnds['boundkey'].to_numpy(), rsk_bnds['lb'].to_numpy(), rsk_bnds['ub'].to_numpy()])
    bnd_con = [bounds['boundkey'].to_numpy(), bounds['lb'].to_numpy(), bounds['ub'].to_numpy()]
    #
    loop_num = max([int(np.ceil((2.0 - to) / to_grad)) + 1, 1])
    for n in range(loop_num):
        to_adj = to_grad * n + to
        turn_con = (w0, to_adj, rho)
        w, is_success, optim_status, _, _ = exec_socp("mosek", u, cov_info, wb, te, lin_con, bnd_con, turn_con=turn_con, return_detail_infos=True)
        if is_success:
            break
    assert is_success
    rtn = conditions[['CalcDate', 'TradeDate', 'Code', alf_col, 'w0']].assign(target_weight=w)
    return rtn