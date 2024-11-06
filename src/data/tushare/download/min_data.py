import tushare as ts

from ..basic import pro
df = ts.pro_bar(ts_code='600000.SH', api = pro , freq='1min', 
                start_date='2024-11-05 00:00:00', 
                end_date='2024-11-05 17:00:00')

df