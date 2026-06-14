"""
Positive elapsed time from a fixed length or from a past timestamp.
"""
from __future__ import annotations
from datetime import datetime , timedelta

__all__ = ['Elapsed' , 'Since']

class TimeDuration:
    """Non-negative elapsed time from either ``duration`` or time since ``since``."""
    def __init__(
        self , duration : float , * , high_precision : bool = False
    ) -> None:
        """
        Args:
            duration: Length of time; ``timedelta`` is converted to seconds.
            since: Start time as POSIX float or ``datetime``; elapsed time is ``now - since``.
        """
        assert duration >= 0 , f"duration must be a positive duration , but got {duration}"
        self.duration = duration
        self.high_precision = high_precision
    def __repr__(self):
        return self.fmtstr
    @property
    def hours(self):
        """Get the duration in hours"""
        return self.duration / 3600
    @property
    def minutes(self):
        """Get the duration in minutes"""
        return self.duration / 60
    @property
    def seconds(self):
        """Get the duration in seconds"""
        return self.duration
    @property
    def days(self):
        """Get the duration in days"""
        return self.duration / 86400
    @property
    def fmtstr(self):
        """Get the duration in a human-readable string"""
        # Calculate time components
        
        # Store components in a dictionary for f-string formatting
        if self.duration < 1:
            return f'{self.duration:.4f} Sec' if self.high_precision else '<1 Sec'
        elif self.duration < 60:
            return f'{self.duration:.4f} Sec' if self.high_precision else f'{self.duration:.1f} Sec'
        else:
            days, remainder = divmod(self.duration, 86400) # 86400 seconds in a day
            hours, remainder = divmod(remainder, 3600)    # 3600 seconds in an hour
            minutes, seconds = divmod(remainder, 60)      # 60 seconds in a minute
        
            fmtstrs = []
            if days > 0:
                fmtstrs.append(f'{days:.0f} Day')
            if hours >= 1:
                fmtstrs.append(f'{hours:.0f} Hour')
            if minutes >= 1:
                fmtstrs.append(f'{minutes:.0f} Min')
            if seconds >= 1:
                fmtstrs.append(f'{seconds:.4f} Sec') if self.high_precision else fmtstrs.append(f'{seconds:.0f} Sec')
            return ' '.join(fmtstrs)

class Elapsed(TimeDuration):
    """Non-negative elapsed time of duration."""
    def __init__(
        self , duration : int | float | timedelta | None = None , / , high_precision : bool = False
    ) -> None:
        """
        Elapsed time of duration.
        Args:
            duration: Length of time; ``timedelta`` is converted to seconds.
        """
        if isinstance(duration , timedelta):
            dur = duration.total_seconds()
        else:
            dur = float(duration) if duration else 0.
        super().__init__(dur, high_precision = high_precision)
    def __repr__(self):
        return self.fmtstr

class Since(TimeDuration):
    """
    Non-negative elapsed time from since to now.
    """
    def __init__(
        self , since : float | datetime | None = None , / ,
        high_precision : bool = False
    ) -> None:
        """
        Since time to now.
        Args:
            since: Start time as POSIX float or ``datetime``; elapsed time is ``now - since``.
        """
        if isinstance(since , datetime):
            dur = (datetime.now() - since).total_seconds()
        else:
            dur = (datetime.now() - datetime.fromtimestamp(since)).total_seconds() if since is not None else 0.
        super().__init__(dur, high_precision = high_precision)