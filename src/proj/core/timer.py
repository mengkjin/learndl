"""Positive elapsed time from a fixed length or from a past timestamp."""

from datetime import datetime , timedelta


class Duration:
    """Non-negative elapsed seconds from either ``duration`` or time since ``since``."""

    def __init__(self , duration : int | float | timedelta | None = None , since : float | datetime | None = None):
        """
        Args:
            duration: Length of time; ``timedelta`` is converted to seconds.
            since: Start time as POSIX float or ``datetime``; elapsed time is ``now - since``.
        """
        assert duration is not None or since is not None , "duration or since must be provided"
        assert duration is None or since is None , f"duration and since cannot be provided at the same time, got duration = {duration} and since = {since}"
        if duration is not None:
            if isinstance(duration , timedelta):
                dur = duration.total_seconds()
            else:
                dur = duration
        elif since is not None:
            if isinstance(since , datetime):
                dur = (datetime.now() - since).total_seconds()
            else:
                dur = (datetime.now() - datetime.fromtimestamp(since)).total_seconds()
        assert dur >= 0 , f"duration must be a positive duration , but got {dur}"
        self.duration = dur
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
            return '<1 Sec'
        elif self.duration < 60:
            return f'{self.duration:.1f} Sec'
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
                fmtstrs.append(f'{seconds:.0f} Sec')
            return ' '.join(fmtstrs)