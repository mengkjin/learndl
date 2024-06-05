from events_system.calendar_util import CALENDAR_UTIL
from collections import OrderedDict


DAY_BEGIN_DATA_TIME = '08:00:00'
DAY_END_TIME = '16:00:00'
QUARTER_TIMES = [
    '09:30:00', '09:45:00', '10:00:00', '10:15:00', '10:30:00', '10:45:00', '11:00:00', '11:15:00', '11:30:00',
    '13:00:00', '13:15:00', '13:30:00', '13:45:00', '14:00:00', '14:15:00', '14:30:00', '14:45:00', '15:00:00',]


class Timeline:

    TRADING_DAY = 0
    NON_TRADING_DAY = 1

    BOD = 2
    EOD = 3
    QOD = 4

    def __init__(self, start_dt_, end_dt_):
        self.m_start_dt = start_dt_
        self.m_end_dt = end_dt_
        #
        self.m_timeline = None
        self.m_cursor = None
        self.crrnt_timer = None
        self._set_timeline()

    def get_datetime_range(self):
        return self.m_timeline[0][0], self.m_timeline[-1][0]

    def _set_timeline(self):
        dates = CALENDAR_UTIL.get_ranged_dates(self.m_start_dt, self.m_end_dt)
        trading_dates = CALENDAR_UTIL.get_ranged_trading_dates(self.m_start_dt, self.m_end_dt)
        self.m_timeline = list()
        timer_counter = 0
        for d in dates:
            day_timers = list()
            if d in trading_dates:
                day_timers.append(
                    (timer_counter, d + ' ' + DAY_BEGIN_DATA_TIME, Timeline.TRADING_DAY, Timeline.BOD)
                )
                timer_counter += 1
                for quarter_tm in QUARTER_TIMES:
                    day_timers.append((timer_counter, d + ' ' + quarter_tm, Timeline.TRADING_DAY, Timeline.QOD))
                    timer_counter += 1
                day_timers.append(
                    (timer_counter, d + ' ' + DAY_END_TIME, Timeline.TRADING_DAY, Timeline.EOD)
                )
                timer_counter += 1
            else:
                day_timers.append(
                    (timer_counter, d + ' ' + DAY_BEGIN_DATA_TIME, Timeline.NON_TRADING_DAY, Timeline.BOD)
                )
                timer_counter += 1
                day_timers.append(
                    (timer_counter, d + ' ' + DAY_END_TIME, Timeline.NON_TRADING_DAY, Timeline.EOD)
                )
                timer_counter += 1
            self.m_timeline.extend(day_timers)
        self.m_cursor = -1

    def move_cursor(self):
        assert self.m_cursor is not None
        self.m_cursor += 1
        if self.m_cursor < len(self.m_timeline):
            self.crrnt_timer = self.m_timeline[self.m_cursor]
            rtn = True
        else:
            self.crrnt_timer = None
            rtn = False
        return rtn

    def get_timer(self):
        return self.crrnt_timer


