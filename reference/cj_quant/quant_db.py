# import 内置模块
import sys
# import 第三方模块
import pandas as pd
# 自有模块
from py_connect import Connector

class DB_Api():
    '''DB_Api类，用于提取云数据库中的数据，目前能够提取的数据有：个股的1分钟K线、个股的财务数据、个股的风格矩阵、
    '''
    def __init__(self,user_ID,timeout=30,opt=0):
        self.__conn = Connector(user_ID, timeout, opt)

    def stock_1min_bar(self, codes: list, date0: str='yyyymmdd', date1: str='yyyymmdd', data_names: list = ['高', '开', '低', '收', '成交量', '成交额', '委买', '委卖', '主买量', '主买额', '主卖量', '主卖额']):
        '''
        提取个股的1分钟K线
        :param codes:个股代码，数据类型为list。
        :param date0: 日期，数据类型为yyyymmdd格式的字符串。
        :param date1: 日期，数据类型为yyyymmdd格式的字符串。当codes是单只股票时。支持提取日期区间内的1分钟K线；若codes为多只股票，只能给定date0，提取当天的所有1分钟K线。date1参数无效，若给定date1则报错。
        :param data_names: 数据类型为list，可为 ['高', '开', '低', '收', '成交量', '成交额', '委买', '委卖', '主买量', '主买额', '主卖量', '主卖额']中的任意一个或多个
        :return: DataFrame
        '''
        func_name, kwargs = self.__get_kwargs(sys._getframe(), locals().items())
        data = self.__conn.get_data_remote(func_name, kwargs)
        data = pd.DataFrame(data)
        if data.empty:
            return data
        data['datetimes'] = data['datetimes'].map(lambda t: pd.Timestamp(t, unit='ms'))
        data['codes'] = self.__codes_fmt_switch(data['codes'] )
        return data

    def stock_financials(self, data_name: str, codes: list, reportDates: str='yyyymmdd', viewDate: str='yyyymmdd', reportType: str = '合并报表')->pd.DataFrame:
        '''
        提取个股的财务数据
        :param data_name: 数据类型为str，可为三大报表中的财务指标，如'净利润(不含少数股东损益)','股东权益合计(不含少数股东权益)','经营活动产生的现金流量净额'等。具体指标名称可通过search方法查询。
        :param codes: 个股代码，数据类型为list。
        :param reportDates: 报告期，数据类型为yyyymmdd格式的字符串。
        :param viewDate: 查看日期，数据类型为yyyymmdd格式的字符串。采用PIT(point in time)模式，即只提取在查看日期之前的最新数据。
        :param reportType: 报表类型，数据类型为str，可为'合并报表','母公司报表'中的任意一个。
        :return: DataFrame
        '''
        func_name, kwargs = self.__get_kwargs(sys._getframe(), locals().items())
        data = self.__conn.get_data_remote(func_name, kwargs)
        data = pd.Series(data)
        return data

    def stock_style_matrix(self, codes: list, date0: str = 'yyyymmdd')->pd.DataFrame:
        '''
        提取指定日期的个股的风格矩阵，包含大小盘、估值、成长、盈利、财务杠杆、机构持仓、动量、反转、beta、波动率、活跃交易等风格因子，
        风格因子为每月初和周初更新，接口返回的是据给定的date0最近日期的最新的风格因子。
        :param codes: 个股代码，数据类型为list。
        :param date0: 日期，数据类型为yyyymmdd格式的字符串。
        :return: DataFrame
        '''
        func_name, kwargs = self.__get_kwargs(sys._getframe(), locals().items())
        data = self.__conn.get_data_remote(func_name, kwargs)
        data = pd.DataFrame(data)
        if data.empty:
            return data
        data['dates'] = data['dates'].map(lambda t: pd.Timestamp(t, unit='ms'))
        data['codes'] = self.__codes_fmt_switch(data['codes'])
        data_matrix = data[['codes','fct_name','fct_value']].pivot(index='codes',columns='fct_name',values='fct_value')
        data_matrix.columns.name = data['dates'].iloc[0]
        data_matrix = data_matrix.reindex(columns = ['大小盘','估值','成长','盈利','财务杠杆','机构持仓','动量','反转','beta','波动率','活跃交易'])
        return data_matrix

    def stock_factors(self, factor_names: list, codes: list, date0: str = 'yyyymmdd', date1: str = 'yyyymmdd')->pd.DataFrame:
        '''
        提取个股的因子数据
        :param factor_names: 因子名称，数据类型为list，具体因子名称可通过search方法查询。
        :param codes: 个股代码，数据类型为list。
        :param date0: 日期，数据类型为yyyymmdd格式的字符串。
        :param date1: 日期，数据类型为yyyymmdd格式的字符串。
        :return: DataFrame
        '''
        func_name, kwargs = self.__get_kwargs(sys._getframe(), locals().items())
        data = self.__conn.get_data_remote(func_name, kwargs)
        data = pd.DataFrame(data)
        if data.empty:
            return data
        data['DATES'] = data['DATES'].map(lambda t: pd.Timestamp(t, unit='ms'))
        data.rename(columns={'DATES':'dates','CODES':'codes'},inplace=True)
        return data

    def search(self,s0:str,if_print=True):
        '''search方法，用于查询stock_financials方法中可用的财务指标名称。
        可以将函数名、财务指标名称等作为查询条件，查询结果会返回所有匹配的结果。
        如：search('净利润')，会返回所有包含‘净利润’的财务指标名称。
        如：search('stock_financials')，会返回所有可通过‘stock_financials’提取的指标。
        :param s0: 查询条件，数据类型为str。
        :param if_print: 是否打印查询结果，数据类型为bool。
        :return: DataFrame
        '''
        func_name, kwargs = self.__get_kwargs(sys._getframe(), locals().items())
        matched_items =  self.__conn.get_data_remote(func_name, kwargs)
        matched_items = pd.DataFrame(matched_items)
        if if_print:
            if not matched_items.empty:
                matched_items_gp = matched_items.groupby('函数名')
                for item in matched_items_gp:
                    print(f"类方法：\n\t {item[0]}")
                    for item2 in item[1].itertuples():
                        if item2[0] == 0:
                            print(f"  *({item2[3]})*")
                        print('|---->' + item2[1])
                    print('*' * 50)
            else:
                print('无匹配')
        return matched_items

    def __codes_fmt_switch(self,codes):
        codes = pd.Series(codes)
        codes_unique = codes.drop_duplicates()
        codes_switched = codes_unique.str.split('.').apply(lambda x: '.'.join(x[::-1]))
        codes = codes.map(dict(zip(codes_unique, codes_switched)))
        return codes

    def __get_kwargs(self,frm,kwargs):
        func_name = frm.f_code.co_name
        kwargs = {key: value for key, value in kwargs if key != 'self'}
        return func_name,kwargs