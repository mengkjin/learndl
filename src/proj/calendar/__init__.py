"""加速版交易日历包（`calendar_new`）。

与 `src.proj.calendar` API 对齐，底层用 ndarray、dict 与 `pandas.Index.get_indexer`
等减少热路径上的 `DataFrame.loc` 开销。业务侧应统一使用本包导出的 `TradeDate`，勿与旧包
`TradeDate` 混用于 `isinstance` 判断。
"""

from .trade_date import TradeDate
from .calendar import CALENDAR, Dates, BJTZ

__all__ = ["TradeDate", "CALENDAR", "Dates", "BJTZ"]
