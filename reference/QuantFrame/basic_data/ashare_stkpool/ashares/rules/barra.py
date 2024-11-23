from barra_model.factor_impl.api import load_barra_data


def get_barra_filter(root_path, scd, ecd):
    rtn = load_barra_data(root_path, 'cne6', scd, ecd)
    rtn = rtn[['CalcDate', 'Code']].assign(has_barra=1)
    return rtn

