import pandas as pd
from basic_src_data.wind_tools.basic import get_stock_list_n_delist


# def _load_raw():
#     df = get_stock_list_n_delist()
#     df['comments'] = ''
#     # patch
#     assert not (df['code'].isin(['000022.SZ', '000043.SZ', '600849.SH'])).any()
#     rtn = df.append(
#         pd.DataFrame([
#             ['000022.SZ', '19930505', '20181231', 'change id to 001872'],
#             ['000043.SZ', '19940928', '20191215', 'change id to 001914'],
#             ['600849.SH', '19940324', '20131231', 'manually added']
#         ],
#          columns=['code', 'list_date', 'delist_date', 'comments'])
#     )
#     assert len(rtn) == len(rtn.code.unique())
#     rtn.sort_values(by=['list_date', 'code'], inplace=True)
#     rtn['list_date'] = rtn['list_date'].apply(func=lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
#     rtn.reset_index(drop=True, inplace=True)
#     return rtn


class IDSys:
    def __init__(self):
        self.df = None

    def _init(self):
        df = get_stock_list_n_delist()
        df['comments'] = ''
        # patch
        assert not (df['code'].isin(['000022.SZ', '000043.SZ', '600849.SH'])).any()
        df = df.append(
            pd.DataFrame([
                ['000022.SZ', '1993-05-05', '2018-12-31', 'change id to 001872'],
                ['000043.SZ', '1994-09-28', '2019-12-15', 'change id to 001914'],
                ['600849.SH', '1994-03-24', '2013-12-31', 'manually added'],
                ['601313.SH', '2012-01-16', '2018-02-28', 'back door listed to 601360']
            ],
             columns=['code', 'list_date', 'delist_date', 'comments'])
        )
        assert len(df) == len(df.code.unique())
        df.sort_values(by=['list_date', 'code'], inplace=True)
        # df['list_date'] = df['list_date'].apply(func=lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
        df.reset_index(drop=True, inplace=True)
        self.df = df
        self.df['sys_id'] = list(range(len(self.df)))

    def get_sys_id(self, code_list_):
        if self.df is None:
            self._init()
        a = pd.merge(pd.DataFrame(code_list_, columns=['code']), self.df, how='left', on=['code'])
        assert pd.notnull(a['sys_id']).all()
        rtn = a.sys_id.values.astype(int)
        return rtn

    def get_tickers(self, sys_id_list_):
        if self.df is None:
            self._init()
        a = pd.merge(pd.DataFrame(sys_id_list_, columns=['sys_id']), self.df, how='left', on=['sys_id'])
        assert pd.notnull(a['code']).all()
        rtn = a['code'].to_numpy()
        return rtn

    def get_all_sys_ids(self):
        if self.df is None:
            self._init()
        return self.df.sys_id.values

    def get_all_tickers(self):
        if self.df is None:
            self._init()
        return self.df.code.values


ID_SYS = IDSys()

