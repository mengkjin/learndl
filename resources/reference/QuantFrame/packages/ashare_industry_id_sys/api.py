from industry_tools.citics import from_en_to_id


def get_sys_id_from_en(industry_type, en_industry_list_):
    assert industry_type == "citics_1"
    rtn = from_en_to_id(en_industry_list_)
    return rtn