from .calendar import CALENDAR
from .stock_info import INFO
from .trade_data import TRADE
from .model_data import MODEL
from .fina_data import INDI

def len_control():
    TRADE.len_control(drop_old = True)
    MODEL.len_control(drop_old = True)
    INDI.len_control(drop_old = True)
