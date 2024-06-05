from .publish_index import load_pub_index_weight_data, calc_pub_index_ret_bias, load_pub_index_level
from .fixed_index import load_fixed_index_weight_data, calc_fixed_index_ret_bias, load_fixed_index_level


def load_index_weight_data(root_path, scd, ecd, index_comp_nm):
    category, index_nm = index_comp_nm.split(':', 1)
    if category == 'publish':
        rtn = load_pub_index_weight_data(root_path, index_nm, scd, ecd)
    elif category == 'fixed':
        rtn = load_fixed_index_weight_data(root_path, index_nm, scd, ecd)
    else:
        assert False, "  errors:>>stk_index_utils>>load_index_weight_data>> category {0} is unknown!".format(category)
    return rtn


def calc_index_ret_bias(root_path, scd, ecd, index_comp_nm):
    category, index_nm = index_comp_nm.split(':', 1)
    if category == 'publish':
        rtn = calc_pub_index_ret_bias(root_path, index_nm, scd, ecd)
    elif category == 'fixed':
        rtn = calc_fixed_index_ret_bias(root_path, index_nm, scd, ecd)
    else:
        assert False, "  errors:>>stk_index_utils>>calc_index_ret_bias>> category {0} is unknown!".format(category)
    return rtn


def load_index_level(root_path, scd, ecd, index_comp_nm):
    category, index_nm = index_comp_nm.split(':', 1)
    if category == 'publish':
        rtn = load_pub_index_level(root_path, index_nm, scd, ecd)
    elif category == 'fixed':
        rtn = load_fixed_index_level(root_path, index_nm, scd, ecd)
    else:
        assert False, "  errors:>>stk_index_utils>>load_index_level>> category {0} is unknown!".format(category)
    return rtn