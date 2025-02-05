# 提取财务报表数据示例

import quant_db

usrId = '' # 请填写您的账号
qdb = quant_db.DB_Api(usrId)


print('*'*50)
print('接口说明:')
help(qdb.stock_factors)

print('示例:')
print('提取因子数据')
codes = ['000001.SZ', '600000.SH'] # 股票代码列表
date0 = '20240101' # 起始日期
date1 = '20241231' # 结束日期
factor_name = ['日频_量价_TCN','1分钟_量价_TCN'] # 因子名称列表
data1 = qdb.stock_factors(factor_name,codes, date0, date1) # 提取因子数据
print(data1)


print('\n使用search辅助查询：')
print('使用search搜索关键字，如”AI因子“，查看包含”AI因子“的因子名称：')
results1 = qdb.search('AI因子', if_print=True)

'''从search结果中解析所有AI因子库中的因子名称'''
fcts_name_all = results1['数据名'].str.split('=', expand=True)
fcts_name_all = fcts_name_all[1].str.split('<', expand=True)
fcts_name_all = fcts_name_all[0]
fcts_name_all = list(fcts_name_all.str.strip())
data2 = qdb.stock_factors(fcts_name_all,codes, date0='20180101',date1 = '20180418') # 提取AI因子数据
print(data2)
