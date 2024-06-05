import pandas as pd
import os


def get_industry_sector_map(industry_type):
    if industry_type == "citics_1":
        sector_file_path = os.path.join(os.path.dirname(__file__), "citics_1_sector_map.csv")
        rtn = pd.read_csv(sector_file_path, encoding="gbk")
        assert not rtn.duplicated(["citics_1"]).any()
    else:
        assert False
    return rtn