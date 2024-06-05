import pandas as pd
from .wind_conn import get_wind_conn


INCOME_WINDFLD_MAP = {
    'revenue': 'TOT_OPER_REV',
    'op_revenue': 'OPER_REV',
    'tot_profit': 'TOT_PROFIT',
    'net_profit': 'NET_PROFIT_EXCL_MIN_INT_INC',
    'op_profit': 'OPER_PROFIT',
    'op_cost': 'LESS_OPER_COST',
    'cost': 'TOT_OPER_COST',
    'net_profit_dnr': 'NET_PROFIT_AFTER_DED_NR_LP',
    'tax': 'INC_TAX',
    'fin_expense': 'LESS_FIN_EXP',
    'sale_expense': 'LESS_SELLING_DIST_EXP',
    'admin_expense': 'LESS_GERL_ADMIN_EXP',
    'rd_expense': 'RD_EXPENSE',
    'op_tax': 'LESS_TAXES_SURCHARGES_OPS',
    'impair_loss_assets': 'LESS_IMPAIR_LOSS_ASSETS',
}

BALANCE_WINDFLD_MAP = {
    'goodwill': 'GOODWILL',
    'equity': 'TOT_SHRHLDR_EQY_EXCL_MIN_INT',
    'liability': 'TOT_LIAB',
    'asset': 'TOT_ASSETS',
    'acc_rcv': 'ACCT_RCV',
    'acc_rec_bill': 'ACCOUNTS_RECEIVABLE_BILL',
    'inventory': 'INVENTORIES',
    'contract_assets': 'CONTRACTUAL_ASSETS',
    'adv_from_cust': 'ADV_FROM_CUST',
    'contract_liab': 'CONTRACT_LIABILITIES',
    'acc_pay': 'ACCT_PAYABLE',
    'accounts_payable': 'ACCOUNTS_PAYABLE',
    'intang_assets': 'INTANG_ASSETS',
    'fix_assets': 'FIX_ASSETS',
    'non_cur_assets': 'TOT_NON_CUR_ASSETS',
    'cur_assets': 'TOT_CUR_ASSETS',
    'cur_liab': 'TOT_CUR_LIAB',
    'non_cur_liab': 'TOT_NON_CUR_LIAB',
    'shrt_trm_brw': 'ST_BORROW',
    'lng_trm_brw': 'LT_BORROW',
    'payable_bonds': 'BONDS_PAYABLE',
    'lease_liab': 'LEASE_LIAB',
    'longterm_deferred_exp': 'LONG_TERM_DEFERRED_EXP',
    'deferred_tax_assets': 'DEFERRED_TAX_ASSETS',
    'non_cur_liab_due_within_1y': 'NON_CUR_LIAB_DUE_WITHIN_1Y',
    'lt_payable': 'LT_PAYABLE',
    'monetary_cap': 'MONETARY_CAP',
    'tradable_fin_assets': 'TRADABLE_FIN_ASSETS',
    'tax_payable': 'TAXES_SURCHARGES_PAYABLE',
    'employ_payable': 'EMPL_BEN_PAYABLE',
    'minority_int': 'MINORITY_INT',
    'held_to_mty_invest': 'HELD_TO_MTY_INVEST',
    'long_term_eqy_invest': 'LONG_TERM_EQY_INVEST',
    'invest_real_estate': 'INVEST_REAL_ESTATE',
    'notes_pay': 'NOTES_PAYABLE',
    'notes_rcv': 'NOTES_RCV',
}

CASHFLOW_WINDFLD_MAP = {
    'op_in_cash': 'STOT_CASH_INFLOWS_OPER_ACT',
    'op_out_cash': 'STOT_CASH_OUTFLOWS_OPER_ACT',
    'op_net_cash': 'NET_CASH_FLOWS_OPER_ACT',
    'inv_in_cash': 'STOT_CASH_INFLOWS_INV_ACT',
    'inv_out_cash': 'STOT_CASH_OUTFLOWS_INV_ACT',
    'inv_net_cash': 'NET_CASH_FLOWS_INV_ACT',
    'fnc_in_cash': 'STOT_CASH_INFLOWS_FNC_ACT',
    'fnc_out_cash': 'STOT_CASH_OUTFLOWS_FNC_ACT',
    'fnc_net_cash': 'NET_CASH_FLOWS_FNC_ACT',
    'net_incr_cash': 'NET_INCR_CASH_CASH_EQU',
    'cash_pay_acq_const_fiolta': 'CASH_PAY_ACQ_CONST_FIOLTA',
    'empl_pay_cash': 'CASH_PAY_BEH_EMPL',
    'cash_end_period': 'CASH_CASH_EQU_END_PERIOD',
    'depr_fa_coga_dpba': 'DEPR_FA_COGA_DPBA',
    'amort_intang_assets': 'AMORT_INTANG_ASSETS',
    'amort_lt_deferred_exp': 'AMORT_LT_DEFERRED_EXP',
    'cash_pay_dist_dpcp_int_exp': 'CASH_PAY_DIST_DPCP_INT_EXP',
}


def load_balance_by_wind(start_ann_date, end_ann_date, query_flds_):
    wind_flds = [BALANCE_WINDFLD_MAP.get(f) for f in query_flds_]
    assert None not in wind_flds
    statement_types = ['408001000', '408004000', '408005000', '408050000']
    conn = get_wind_conn()
    sql = "select S_INFO_WINDCODE, ACTUAL_ANN_DT, REPORT_PERIOD, STATEMENT_TYPE, {0} from AShareBalanceSheet " \
          "where ACTUAL_ANN_DT between '{1}' and '{2}' and STATEMENT_TYPE in ({3}) " \
          "order by ACTUAL_ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, STATEMENT_TYPE".format(
        ",".join(wind_flds),
        start_ann_date.replace('-', ''), end_ann_date.replace('-', ''),
        ",".join(statement_types)
    )
    df = pd.read_sql(sql, conn)
    df.columns = df.columns.str.upper()
    df.rename(columns={"S_INFO_WINDCODE": "Code", "ACTUAL_ANN_DT": "AnnDate", "REPORT_PERIOD": "report_period",
                       "STATEMENT_TYPE": "statement_type"}, errors="raise", inplace=True)
    df.rename(columns=dict(zip(wind_flds, query_flds_)), inplace=True)
    df.drop_duplicates(subset=['Code', 'AnnDate', 'report_period'], keep='last', inplace=True)
    if 'inventory' in df.columns:
        df['inventory'].fillna(0.0, inplace=True)
    if 'contract_assets' in df.columns:
        df['contract_assets'].fillna(0.0, inplace=True)
    if 'adv_from_cust' in df.columns:
        df['adv_from_cust'].fillna(0.0, inplace=True)
    if 'contract_liab' in df.columns:
        df['contract_liab'].fillna(0.0, inplace=True)
    if 'lng_trm_brw' in df.columns:
        df['lng_trm_brw'].fillna(0.0, inplace=True)
    if 'payable_bonds' in df.columns:
        df['payable_bonds'].fillna(0.0, inplace=True)
    if 'shrt_trm_brw' in df.columns:
        df['shrt_trm_brw'].fillna(0.0, inplace=True)
    if 'lease_liab' in df.columns:
        df['lease_liab'].fillna(0.0, inplace=True)
    if 'goodwill' in df.columns:
        df['goodwill'] = df['goodwill'].fillna(0.0)
    if 'longterm_deferred_exp' in df.columns:
        df['longterm_deferred_exp'].fillna(0.0, inplace=True)
    if 'deferred_tax_assets' in df.columns:
        df['deferred_tax_assets'].fillna(0.0, inplace=True)
    if 'non_cur_liab_due_within_1y' in df.columns:
        df['non_cur_liab_due_within_1y'].fillna(0.0, inplace=True)
    if 'lt_payable' in df.columns:
        df['lt_payable'].fillna(0.0, inplace=True)
    if 'monetary_cap' in df.columns:
        df['monetary_cap'].fillna(0.0, inplace=True)
    if 'tradable_fin_assets' in df.columns:
        df['tradable_fin_assets'].fillna(0.0, inplace=True)
    if 'tax_payable' in df.columns:
        df['tax_payable'].fillna(0.0, inplace=True)
    if 'employ_payable' in df.columns:
        df['employ_payable'].fillna(0.0, inplace=True)
    if 'minority_int' in df.columns:
        df['minority_int'].fillna(0.0, inplace=True)
    if 'held_to_mty_invest' in df.columns:
        df['held_to_mty_invest'].fillna(0.0, inplace=True)
    if 'long_term_eqy_invest' in df.columns:
        df['long_term_eqy_invest'].fillna(0.0, inplace=True)
    if 'invest_real_estate' in df.columns:
        df['invest_real_estate'].fillna(0.0, inplace=True)
    #
    df = df.loc[
        (df['Code'].str[0] != 'A') & df['report_period'].str[4:8].isin(['0331', '0630', '0930', '1231']),
        ['Code', 'AnnDate', 'report_period'] + query_flds_
    ].copy()
    df['AnnDate'] = df['AnnDate'].str[:4] + '-' + df['AnnDate'].str[4:6] + '-' + df['AnnDate'].str[6:]
    return df


def load_income_by_wind(start_ann_date, end_ann_date, query_flds_):
    wind_flds = [INCOME_WINDFLD_MAP.get(f) for f in query_flds_]
    assert None not in wind_flds
    statement_types = ['408001000', '408004000', '408005000', '408050000']
    conn = get_wind_conn()
    sql = "select S_INFO_WINDCODE, ACTUAL_ANN_DT, REPORT_PERIOD, STATEMENT_TYPE, {0} from AShareIncome " \
          "where ACTUAL_ANN_DT between '{1}' and '{2}' and STATEMENT_TYPE in ({3}) " \
          "order by ACTUAL_ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, STATEMENT_TYPE".format(
        ",".join(wind_flds),
        start_ann_date.replace('-', ''), end_ann_date.replace('-', ''),
        ",".join(statement_types)
    )
    df = pd.read_sql(sql, conn)
    df.columns = df.columns.str.upper()
    df.rename(columns={"S_INFO_WINDCODE": "Code", "ACTUAL_ANN_DT": "AnnDate", "REPORT_PERIOD": "report_period",
                       "STATEMENT_TYPE": "statement_type"}, errors="raise", inplace=True)
    df.rename(columns=dict(zip(wind_flds, query_flds_)), inplace=True)
    df.drop_duplicates(subset=['Code', 'AnnDate', 'report_period'], keep='last', inplace=True)
    df = df.loc[
        (df['Code'].str[0] != 'A') & df['report_period'].str[4:8].isin(['0331', '0630', '0930', '1231']),
        ['Code', 'AnnDate', 'report_period'] + query_flds_
    ].copy()
    df['AnnDate'] = df['AnnDate'].str[:4] + '-' + df['AnnDate'].str[4:6] + '-' + df['AnnDate'].str[6:]
    if 'fin_expense' in df.columns:
        df['fin_expense'].fillna(0.0, inplace=True)
    if 'tax' in df.columns:
        df['tax'].fillna(0.0, inplace=True)
    if 'sale_expense' in df.columns:
        df['sale_expense'].fillna(0.0, inplace=True)
    if 'admin_expense' in df.columns:
        df['admin_expense'].fillna(0.0, inplace=True)
    if 'op_tax' in df.columns:
        df['op_tax'].fillna(0.0, inplace=True)
    if 'impair_loss_assets' in df.columns:
        df['impair_loss_assets'].fillna(0.0, inplace=True)
    return df


def load_cashflow_by_wind(start_ann_date, end_ann_date, query_flds_):
    wind_flds = [CASHFLOW_WINDFLD_MAP.get(f) for f in query_flds_]
    assert None not in wind_flds
    statement_types = ['408001000', '408004000', '408005000', '408050000']
    conn = get_wind_conn()
    sql = "select S_INFO_WINDCODE, ACTUAL_ANN_DT, REPORT_PERIOD, STATEMENT_TYPE, {0} from AShareCashFlow " \
          "where ACTUAL_ANN_DT between '{1}' and '{2}' and STATEMENT_TYPE in ({3}) " \
          "order by ACTUAL_ANN_DT, S_INFO_WINDCODE, REPORT_PERIOD, STATEMENT_TYPE".format(
        ",".join(wind_flds),
        start_ann_date.replace('-', ''), end_ann_date.replace('-', ''),
        ",".join(statement_types)
    )
    df = pd.read_sql(sql, conn)
    df.columns = df.columns.str.upper()
    df.rename(columns={"S_INFO_WINDCODE": "Code", "ACTUAL_ANN_DT": "AnnDate", "REPORT_PERIOD": "report_period",
                       "STATEMENT_TYPE": "statement_type"}, errors="raise", inplace=True)
    df.rename(columns=dict(zip(wind_flds, query_flds_)), inplace=True)
    df.drop_duplicates(subset=['Code', 'AnnDate', 'report_period'], keep='last', inplace=True)
    df = df[['Code', 'AnnDate', 'report_period'] + query_flds_]
    df = df[df['Code'].str[0] != 'A']
    df = df[df['report_period'].str[4:8].isin(['0331', '0630', '0930', '1231'])].copy()
    df['AnnDate'] = df['AnnDate'].str[:4] + '-' + df['AnnDate'].str[4:6] + '-' + df['AnnDate'].str[6:]
    if 'cash_pay_acq_const_fiolta' in df.columns:
        df['cash_pay_acq_const_fiolta'].fillna(0.0, inplace=True)
    if 'empl_pay_cash' in df.columns:
        df['empl_pay_cash'].fillna(0.0, inplace=True)
    if 'cash_end_period' in df.columns:
        df['cash_end_period'].fillna(0.0, inplace=True)
    if 'depr_fa_coga_dpba' in df.columns:
        df['depr_fa_coga_dpba'].fillna(0.0, inplace=True)
    if 'amort_intang_assets' in df.columns:
        df['amort_intang_assets'].fillna(0.0, inplace=True)
    if 'amort_lt_deferred_exp' in df.columns:
        df['amort_lt_deferred_exp'].fillna(0.0, inplace=True)
    if 'cash_pay_dist_dpcp_int_exp' in df.columns:
        df['cash_pay_dist_dpcp_int_exp'].fillna(0.0, inplace=True)
    return df


def load_notice_data(start_ann_date, end_ann_date):
    conn = get_wind_conn()
    sql = "select S_INFO_WINDCODE, S_PROFITNOTICE_DATE, S_PROFITNOTICE_PERIOD, " \
          "S_PROFITNOTICE_CHANGEMAX, S_PROFITNOTICE_CHANGEMIN, " \
          "S_PROFITNOTICE_NETPROFITMIN, S_PROFITNOTICE_NETPROFITMAX, S_PROFITNOTICE_STYLE " \
          "from AShareProfitNotice where S_PROFITNOTICE_DATE between '{0}' and '{1}' " \
          "order by S_PROFITNOTICE_DATE".format(start_ann_date.replace('-', ''), end_ann_date.replace('-', ''))
    notice_df = pd.read_sql(sql, conn)
    notice_df.columns = notice_df.columns.str.upper()
    notice_df.rename(
        columns={"S_INFO_WINDCODE": "Code", "S_PROFITNOTICE_DATE": "AnnDate", "S_PROFITNOTICE_PERIOD": "report_period",
                 "S_PROFITNOTICE_CHANGEMAX": "rate_max", "S_PROFITNOTICE_CHANGEMIN": "rate_min",
                 "S_PROFITNOTICE_NETPROFITMIN": "amt_min", "S_PROFITNOTICE_NETPROFITMAX": "amt_max",
                 "S_PROFITNOTICE_STYLE": "style"},
              errors="raise", inplace=True)
    notice_df['AnnDate'] = notice_df['AnnDate'].apply(func=lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    notice_df['amt_max'] = notice_df['amt_max'] * 10000.0
    notice_df['amt_min'] = notice_df['amt_min'] * 10000.0
    notice_df['rate_max'] = notice_df['rate_max'] / 100.0
    notice_df['rate_min'] = notice_df['rate_min'] / 100.0
    notice_df.dropna(subset=['rate_max', 'rate_min', 'amt_min', 'amt_max'], how='all', inplace=True)
    notice_df.drop_duplicates(subset=['Code', 'AnnDate', 'report_period'], inplace=True)
    notice_df['style'] = notice_df['style'].fillna(value=-1)
    notice_df['last_yoy_sign'] = -(notice_df['style'].astype(int).isin([454007000, 454004000]) * 2 - 1)
    return notice_df


EXPR_WINDFLD_MAP = {
    'op_revenue': 'OPER_REV',
    'tot_profit': 'TOT_PROFIT',
    'net_profit': 'NET_PROFIT_EXCL_MIN_INT_INC',
    'op_profit': 'OPER_PROFIT',
    'asset': 'TOT_ASSETS',
    'equity': 'TOT_SHRHLDR_EQY_EXCL_MIN_INT'
}


def load_express_data(start_ann_date, end_ann_date, query_flds_):
    flds = [EXPR_WINDFLD_MAP.get(f) for f in query_flds_]
    conn = get_wind_conn()
    sql = "select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, {0} from AShareProfitExpress " \
          "where ANN_DT between '{1}' and '{2}' order by ANN_DT, S_INFO_WINDCODE".format(
           ', '.join(flds),
           start_ann_date.replace('-', ''), end_ann_date.replace('-', ''))
    df = pd.read_sql(sql, conn)
    df.columns = df.columns.str.upper()
    df.rename(columns={"S_INFO_WINDCODE": "Code", "ANN_DT": "AnnDate", "REPORT_PERIOD": "report_period"},
              errors="raise", inplace=True)
    df.rename(columns=dict(zip(flds, query_flds_)), inplace=True)
    df['AnnDate'] = df['AnnDate'].apply(func=lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    return df