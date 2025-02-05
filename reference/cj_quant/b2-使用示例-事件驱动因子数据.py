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
date0 = '20220101' # 起始日期
date1 = '20221231' # 结束日期
factor_name = ['高管增持事件因子','负向业绩预告事件因子'] # 因子名称列表
data1 = qdb.stock_factors(factor_name,codes, date0, date1) # 提取因子数据
print(data1)


print('\n使用search辅助查询：')
print('使用search搜索关键字，如”事件驱动因子库“，查看包含”事件驱动因子库“的因子名称：')
results1 = qdb.search('事件驱动因子库', if_print=True)

'''从search结果中解析所有事件驱动因子库中的因子名称'''
fcts_name_all = results1['数据名'].str.split('=', expand=True)
fcts_name_all = fcts_name_all[1].str.split('<', expand=True)
fcts_name_all = fcts_name_all[0]
fcts_name_all = list(fcts_name_all.str.strip())
data2 = qdb.stock_factors(fcts_name_all,codes, date0='20220101',date1 = '20221231') # 提取所有事件驱动因子数据
print(data2)
