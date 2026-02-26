import yaml
import os


def get_configs(config_file):
    with open(config_file) as file:
        configs = yaml.safe_load(file.read())
    return configs


def get_build_in_configs(port_cate):
    config_file = os.path.join(os.path.dirname(__file__), "configs\\{0}.yaml".format(port_cate))
    rtn = get_configs(config_file)
    rtn['port_cate'] = port_cate
    return rtn
