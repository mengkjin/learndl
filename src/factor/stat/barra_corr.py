import pandas as pd
from barra_model.factor_impl.api import load_barra_data


def calc_barra_corr(root_path, factor_val_df, barra_type):
    barra_data = load_barra_data(root_path, barra_type, factor_val_df["CalcDate"].min(), factor_val_df["CalcDate"].max())
    barra_data = barra_data.filter(regex="CalcDate|Code|STYLE", axis=1)
    barra_col = barra_data.columns.drop(["CalcDate", "Code"]).tolist()
    factor_col = factor_val_df.columns.drop(["CalcDate", "Code"]).tolist()
    factor_barra_val = pd.merge(factor_val_df, barra_data, on=["CalcDate", "Code"], how="inner")
    #
    factor_barra_corr = factor_barra_val.set_index(["CalcDate", "Code"]).groupby(["CalcDate"]).apply(
        lambda x: x.corr(method="spearman").loc[factor_col, barra_col])
    factor_barra_corr.index.rename("factor_name", level=1, inplace=True)
    factor_barra_corr.reset_index(drop=False, inplace=True)
    return factor_barra_corr