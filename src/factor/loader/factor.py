from ...data import DATAVENDOR

def random(start_dt = 20240101 , end_dt = 20240531 , step = 5 , nfactor = 2):
    return DATAVENDOR.random_factor(start_dt , end_dt , step , nfactor).to_dataframe()

