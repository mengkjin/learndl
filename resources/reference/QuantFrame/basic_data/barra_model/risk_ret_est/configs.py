import os


def get_risk_ret_path(root_path, barra_type):
    path = os.sep.join([root_path, "barra_data", "risk_ret", barra_type])
    return path


def get_special_ret_path(root_path, barra_type):
    path = os.sep.join([root_path, "barra_data", "risk_ret", barra_type, "special_ret"])
    return path