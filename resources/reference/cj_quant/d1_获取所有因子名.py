import pandas as pd

import quant_db

usrId = '' # 请填写您的账号
qdb = quant_db.DB_Api(usrId)
def get_all_factor_names(qdb):
    results1 = qdb.search('stock_factors', if_print=False)

    '''从search结果中解析因子库中的因子名称'''
    fcts_name_all = results1['数据名'].str.split('=', expand=True)
    fcts_name_all = fcts_name_all[1].str.split('<', expand=True)
    fct_name = fcts_name_all[0].str.strip()

    fct_type = fcts_name_all[1].str.split('，', expand=True)[1]
    fcts_name_all = pd.concat([fct_name, fct_type,results1['备注']], axis=1)
    fcts_name_all.columns = ['因子名称', '因子类型', '因子描述']
    return fcts_name_all

if __name__ == '__main__':
    fcts_name_all = get_all_factor_names(qdb) # 用上面的函数获取所有因子名称
    factor_names = fcts_name_all['因子名称'].tolist()
    data = qdb.stock_factors(factor_names, ['000001.SZ','600000.SH'], date0='20230101', date1='20231231') # 其中月度因子仅月末有值，日度因子为日度更新
    print(data)