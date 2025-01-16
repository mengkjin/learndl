# 提取1分钟K线数据示例

import quant_db

usrId = '' # 请填写您的账号
qdb = quant_db.DB_Api(usrId)

print('*'*50)
print('接口说明:')
help(qdb.stock_1min_bar)

print('示例:')
print('1.1 单只股票，支持提取日期区间内的1分钟K线')
codes = ['000001.SZ']
date0 = '20220101'
date1 = '20220131'
Kbar1 = qdb.stock_1min_bar(codes, date0, date1)
print(Kbar1)

print('1.2 多只股票,支持提取某一天的1分钟K线')
codes = ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ']
date0 = '20220106'
Kbar2 = qdb.stock_1min_bar(codes, date0)
print(Kbar2)

print('1.3 data_names参数，用于提取指定的数据列')
codes = ['000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ']
date0 = '20220106'
data_names = ['开', '收', '成交量']
Kbar3 = qdb.stock_1min_bar(codes, date0, data_names=data_names)
print(Kbar3)

date1 = '20220131'
Kbar4 = qdb.stock_1min_bar(['000001.SZ'], date0, date1, data_names=data_names) # 单只股票
print(Kbar4)

