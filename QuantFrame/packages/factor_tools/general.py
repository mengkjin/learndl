import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def onehotize(df):
    enc = OneHotEncoder(handle_unknown='error')
    enc.fit(df)
    ind_factors = enc.transform(df).toarray()
    rtn = pd.DataFrame(ind_factors, index=df.index, columns=enc.categories_[0])
    return rtn