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
factor_name = ['特异率','净利润率'] # 因子名称列表
data1 = qdb.stock_factors(factor_name,codes, date0, date1) # 提取因子数据
print(data1)


print('\n使用search辅助查询：')
print('使用search搜索关键字，如”净利润“，查看包含”净利润“的因子名称：')
results1 = qdb.search('净利润', if_print=True)

print('\n使用search接口搜索函数名stock_factors，可查看所有支持的因子：')
results2 = qdb.search('stock_factors', if_print=True)

print('\n使用search接口搜索因子类型名，如："高频因子库",可查看所有同类因子：')
results3 = qdb.search('高频因子库', if_print=True)
