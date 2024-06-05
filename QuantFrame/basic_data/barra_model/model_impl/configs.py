import os


def get_model_path(root_path, barra_type):
    path = os.sep.join([root_path, "barra_data", "models", barra_type])
    return path