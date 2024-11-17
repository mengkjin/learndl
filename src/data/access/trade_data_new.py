import pandas as pd
import numpy as np

from typing import Literal
from ...func.singleton import singleton

@singleton
class TradeDataAccess:
    def get_income_statement(self, date: int, field: str, freq: Literal['quarterly', 'yearly'] = 'yearly') -> pd.Series:
        """获取利润表数据
        Args:
            date: 日期
            field: 字段名
            freq: 频率, quarterly-季度, yearly-年度
        Returns:
            pd.Series: index为股票代码的Series
        """
        # TODO: 实现数据获取逻辑
        return pd.Series()

    def get_balance_sheet(self, date: int, field: str) -> pd.Series:
        """获取资产负债表数据
        Args:
            date: 日期
            field: 字段名
        Returns:
            pd.Series: index为股票代码的Series
        """
        # TODO: 实现数据获取逻辑
        return pd.Series()

    def get_cashflow(self, date: int, field: str, freq: Literal['quarterly', 'yearly'] = 'yearly') -> pd.Series:
        """获取现金流量表数据
        Args:
            date: 日期
            field: 字段名
            freq: 频率, quarterly-季度, yearly-年度
        Returns:
            pd.Series: index为股票代码的Series
        """
        # TODO: 实现数据获取逻辑
        return pd.Series()

    def get_market_value(self, date: int) -> pd.Series:
        """获取市值数据
        Args:
            date: 日期
        Returns:
            pd.Series: index为股票代码的Series
        """
        # TODO: 实现数据获取逻辑
        return pd.Series()

    def get_industry(self, date: int) -> pd.Series:
        """获取行业分类数据
        Args:
            date: 日期
        Returns:
            pd.Series: index为股票代码的Series
        """
        # TODO: 实现数据获取逻辑
        return pd.Series()

    def get_consensus_forecast(self, date: int, field: str) -> pd.Series:
        """获取一致预期数据
        Args:
            date: 日期
            field: 字段名
        Returns:
            pd.Series: index为股票代码的Series
        """
        # TODO: 实现数据获取逻辑
        return pd.Series()

    def get_dividend(self, date: int) -> pd.Series:
        """获取股息数据
        Args:
            date: 日期
        Returns:
            pd.Series: index为股票代码的Series
        """
        # TODO: 实现数据获取逻辑
        return pd.Series()