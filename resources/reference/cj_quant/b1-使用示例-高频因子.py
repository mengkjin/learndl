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
factor_name = ['驱动大笔相关性_10','小笔错位相关性_20'] # 因子名称列表
print("高频因子数据为日频更新，以下为提取2022年全年 '驱动大笔相关性_10'、'小笔错位相关性_20' 两个的数据的例子：")
data1 = qdb.stock_factors(factor_name,codes, date0='20220101',date1 = '20221231') # 提取因子数据
print(data1)

print("高频因子数据为日频更新，以下为提取2022年全年所有高频因子数据的例子：")
#通过search函数获取所有日频更新的高频因子名称
fcts_name_all = qdb.search('高频因子库（日频更新）', if_print=False)
fcts_name_all = fcts_name_all['数据名'].str.split('=', expand=True)
fcts_name_all = fcts_name_all[1].str.split('<', expand=True)
fcts_name_all = fcts_name_all[0]
fcts_name_all = list(fcts_name_all.str.strip())
data2 = qdb.stock_factors(fcts_name_all,codes, date0='20220101',date1 = '20221231') # 提取因子数据
print(data2)

