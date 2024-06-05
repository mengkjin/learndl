import itertools


def get_last_report_periods(report_period, n_=1):
    assert n_ >= 0
    if n_ == 0:
        print("  warning::financial_tools>>report_periods>>if n is zero, meaning keep the given report_period as the result.")
    rtn = [report_period]
    for i in range(n_):
        report_period = get_last_rpt_prd(report_period)
        rtn.append(report_period)
    return rtn


def get_last_rpt_prd(report_period):
    report_y, report_md = report_period[:4], report_period[4:]
    if report_md == '0331':
        rtn = str(int(report_y) - 1) + '1231'
    elif report_md == "0630":
        rtn = report_y + '0331'
    elif report_md == "0930":
        rtn = report_y + '0630'
    elif report_md == "1231":
        rtn = report_y + '0930'
    else:
        assert False
    return rtn


def get_last_yoy_prd(report_period):
    report_y, report_md = report_period[:4], report_period[4:]
    rtn = str(int(report_y) - 1) + report_md
    return rtn


def get_next_rpt_prd(report_period):
    report_period = str(report_period)
    report_y, report_md = report_period[:4], report_period[4:]
    if report_md == '0331':
        rtn = report_y + '0630'
    elif report_md == '0630':
        rtn = report_y + '0930'
    elif report_md == '0930':
        rtn = report_y + '1231'
    elif report_md == '1231':
        rtn = str(int(report_y) + 1) + '0331'
    else:
        assert False
    return rtn


def get_report_period_in_range(lft_rpt_prd_, rht_rpt_prd_):
    lft_y, rht_y = int(lft_rpt_prd_[:4]), int(rht_rpt_prd_[:4])
    rtn = [str(y) + md for y, md in itertools.product(range(lft_y, rht_y + 1), ['0331', '0630', '0930', '1231'])
           if lft_rpt_prd_ <= str(y) + md <= rht_rpt_prd_]
    return rtn


def get_report_period_dist(lft_rpt_prd_, rht_rpt_prd_):
    rht_y, rht_m = int(rht_rpt_prd_[:4]), int(rht_rpt_prd_[4:6])
    lft_y, lft_m = int(lft_rpt_prd_[:4]), int(lft_rpt_prd_[4:6])
    y_dist = rht_y - lft_y
    m_dist, _ = divmod(rht_m - lft_m, 3)
    rtn = y_dist * 4 + m_dist
    return rtn


def get_statement_rptprd_range(date_, freq="q"):
    assert freq == 'q'
    y, last_y = date_[:4], str(int(date_[:4]) - 1)
    m = int(date_[5:7])
    if m == 5 or m == 6:
        rtn = y + '0331', y + '0331'
    elif m == 7 or m == 8:
        rtn = y + '0331', y + '0630'
    elif m == 9:
        rtn = y + '0630', y + '0630'
    elif m == 10:
        rtn = y + '0630', y + '0930'
    elif m == 11 or m == 12:
        rtn = y + '0930', y + '0930'
    elif m == 1 or m == 2 or m == 3:
        rtn = last_y + '0930', last_y + '1231'
    elif m == 4:
        rtn = last_y + '0930', y + '0331'
    else:
        assert False
    return rtn[0], rtn[1]


def get_ntcstm_rptprd_range(date_):
    raise NotImplementedError


def get_valid_nes_ann_rptprd_range(date_):
    y, last_y = date_[:4], str(int(date_[:4]) - 1)
    m = int(date_[5:7])
    if m == 5 or m == 6:
        rtn_ = y + '0630', y + '0630'
    elif m == 7 or m == 8:
        rtn_ = y + '0630', y + '0930'
    elif m == 9:
        rtn_ = y + '0930', y + '0930'
    elif m == 10:
        rtn_ = y + '0930', y + '1231'
    elif m == 11 or m == 12:
        rtn_ = y + '1231', y + '1231'
    elif m == 1 or m == 2 or m == 3:
        rtn_ = last_y + '1231', y + '0331'
    elif m == 4:
        rtn_ = last_y + '1231', y + '0630'
    else:
        assert False
    return int(rtn_[0]), int(rtn_[1])

