import numpy as np
import pandas as pd


def calc_is_limited(src_data, fld_list=("is_limit_eod",), keep_origin=False):
    # TODO: not deal with the case of the first day and first week's up low limit
    assert pd.Index(['CalcDate', 'Code', 'close_price', 'prev_close', 'high_price', 'low_price']).difference(src_data.columns).empty
    if keep_origin:
        data = src_data.copy()
    else:
        data = src_data[["CalcDate", "Code"]].copy()
    kechuangban_flg = data['Code'].str[:3] == '688'
    chuangyeban_flg = data['Code'].str[1] == '3'
    chuangyeban_zhucezhi_flg = data['CalcDate'] >= '2020-06-15'
    other_flg = (~kechuangban_flg) & (~chuangyeban_flg)
    #
    if "is_limit_eod" in fld_list:
        data['is_limit_eod'] = 0
        abs_exceed_10 = np.abs(src_data['close_price'] / src_data['prev_close'] - 1.0) > 0.098
        abs_exceed_20 = np.abs(src_data['close_price'] / src_data['prev_close'] - 1.0) > 0.198
        hgh_cls_same = (src_data['high_price'] - 0.0001) < src_data['close_price']
        low_cls_same = (src_data['low_price'] + 0.0001) > src_data['close_price']
        up_limit_by_10_flg = abs_exceed_10 & hgh_cls_same
        lw_limit_by_10_flg = abs_exceed_10 & low_cls_same
        up_limit_by_20_flg = abs_exceed_20 & hgh_cls_same
        lw_limit_by_20_flg = abs_exceed_20 & low_cls_same
        data.loc[other_flg & up_limit_by_10_flg, 'is_limit_eod'] = 1
        data.loc[other_flg & lw_limit_by_10_flg, 'is_limit_eod'] = -1
        data.loc[kechuangban_flg & up_limit_by_20_flg, 'is_limit_eod'] = 1
        data.loc[kechuangban_flg & lw_limit_by_20_flg, 'is_limit_eod'] = -1
        data.loc[chuangyeban_flg & (~chuangyeban_zhucezhi_flg) & up_limit_by_10_flg, 'is_limit_eod'] = 1
        data.loc[chuangyeban_flg & (~chuangyeban_zhucezhi_flg) & lw_limit_by_10_flg, 'is_limit_eod'] = -1
        data.loc[chuangyeban_flg & chuangyeban_zhucezhi_flg & up_limit_by_20_flg, 'is_limit_eod'] = 1
        data.loc[chuangyeban_flg & chuangyeban_zhucezhi_flg & lw_limit_by_20_flg, 'is_limit_eod'] = -1
    #
    if "hit_up_limit" in fld_list:
        data["hit_up_limit"] = 0
        hit_up_limit_by_10_flg = (src_data['high_price'] / src_data['prev_close'] - 1.0) > 0.098
        hit_up_limit_by_20_flg = (src_data['high_price'] / src_data['prev_close'] - 1.0) > 0.198
        data.loc[other_flg & hit_up_limit_by_10_flg, 'hit_up_limit'] = 1
        data.loc[kechuangban_flg & hit_up_limit_by_20_flg, 'hit_up_limit'] = 1
        data.loc[chuangyeban_flg & (~chuangyeban_zhucezhi_flg) & hit_up_limit_by_10_flg, 'hit_up_limit'] = 1
        data.loc[chuangyeban_flg & chuangyeban_zhucezhi_flg & hit_up_limit_by_20_flg, 'hit_up_limit'] = 1
    #
    if "hit_low_limit" in fld_list:
        data["hit_low_limit"] = 0
        hit_lw_limit_by_10_flg = (src_data['low_price'] / src_data['prev_close'] - 1.0) < -0.098
        hit_lw_limit_by_20_flg = (src_data['low_price'] / src_data['prev_close'] - 1.0) < -0.198
        data.loc[other_flg & hit_lw_limit_by_10_flg, 'hit_low_limit'] = 1
        data.loc[kechuangban_flg & hit_lw_limit_by_20_flg, 'hit_low_limit'] = 1
        data.loc[chuangyeban_flg & (~chuangyeban_zhucezhi_flg) & hit_lw_limit_by_10_flg, 'hit_low_limit'] = 1
        data.loc[chuangyeban_flg & chuangyeban_zhucezhi_flg & hit_lw_limit_by_20_flg, 'hit_low_limit'] = 1
    return data