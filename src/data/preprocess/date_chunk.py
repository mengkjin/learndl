"""
Calendar-year (or N-year) date windows for large ``DataBlock`` preprocess jobs.

``PreProcessor.DateChunkYears`` > 0 splits a ``[start, end]`` yyyymmdd span so
``pre_process`` never loads the full history at once.  Each window is inclusive.
"""
from __future__ import annotations


def calendar_year_spans(start : int , end : int , years : int) -> list[tuple[int , int]]:
    """
    Split ``[start, end]`` into consecutive calendar-year groups.

    Parameters
    ----------
    start, end : int
        Inclusive dates as ``yyyymmdd``.
    years : int
        Number of calendar years per chunk. ``<= 0`` returns the original span.

    Returns
    -------
    list[tuple[int, int]]
        Inclusive ``(chunk_start, chunk_end)`` windows covering ``[start, end]``.
    """
    if years <= 0 or start > end:
        return [] if start > end else [(int(start) , int(end))]
    start , end = int(start) , int(end)
    y0 = start // 10000
    y1 = end // 10000
    spans : list[tuple[int , int]] = []
    year = y0
    while year <= y1:
        chunk_last = min(year + years - 1 , y1)
        lo = start if year == y0 else year * 10000 + 101
        hi = end if chunk_last == y1 else chunk_last * 10000 + 1231
        if lo <= hi:
            spans.append((lo , hi))
        year = chunk_last + 1
    return spans
